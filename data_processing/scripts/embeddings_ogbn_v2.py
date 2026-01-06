import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
from tqdm import tqdm
from huggingface_hub import login
import os
import gc
from dotenv import load_dotenv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except:
        nltk.download('punkt_tab')  # For newer NLTK versions

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Model configurations
MODEL_CONFIGS = {
    # LLaMA models
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8B": "meta-llama/Llama-3.1-8B",

    # Qwen models
    "qwen-3.3-32b": "Qwen/Qwen2.5-32B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-3-32b-awq": "Qwen/Qwen3-32B-AWQ",
    "qwen-3-8B-embedding"  : "Qwen/Qwen3-Embedding-8B",

    # DeepSeek models
    "deepseek-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


class CUDAOptimizedEmbeddingGenerator:
    """CUDA-optimized embedding generator for citation network papers."""

    def __init__(self, model_name: str = "llama-3.2-3b", device: Optional[str] = None,
                 embedding_mode: str = "title", remove_stopwords: bool = False,
                 stopwords_lang: str = "english"):
        """
        Initialize CUDA-optimized embedding generator.

        Args:
            model_name: Model identifier (key from MODEL_CONFIGS or HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            embedding_mode: What to embed ('title', 'abstract', 'authors', 'combined', 'title_abstract')
            remove_stopwords: Whether to remove stopwords before embedding
            stopwords_lang: Language for stopwords (default: 'english')
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Resolve model name
        self.model_name = MODEL_CONFIGS.get(model_name, model_name)
        self.model_type = self._detect_model_type(self.model_name)
        self.embedding_mode = embedding_mode
        self.remove_stopwords = remove_stopwords
        self.stopwords_lang = stopwords_lang

        self.device = self._setup_device(device)
        print("Running on:", self.device)
        self.tokenizer = None
        self.model = None

        # CUDA optimization settings
        self.use_cuda = self.device == "cuda"

        # Memory optimization settings
        self.max_batch_size = self._get_optimal_batch_size()
        self.mixed_precision = False
        self.gradient_checkpointing = False

        # Initialize CUDA optimization

        # Initialize stopwords if needed
        if self.remove_stopwords:
            try:
                self.stop_words = set(stopwords.words(self.stopwords_lang))
                self.logger.info(f"Initialized stopword removal for language: {self.stopwords_lang}")
                self.logger.info(f"Number of stopwords: {len(self.stop_words)}")
            except Exception as e:
                self.logger.warning(f"Could not load NLTK stopwords: {e}")
                self.logger.warning("Using basic English stopwords as fallback")
                # Fallback to basic stopwords if NLTK fails
                self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
                                   'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                                   'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                                   'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                                   'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                                   'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                                   'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                                   'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                                   'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                                   'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                                   'further', 'then', 'once'}
        else:
            self.stop_words = None

        self.logger.info(f"Initialized with model: {self.model_name}")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Embedding mode: {self.embedding_mode}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"CUDA optimizations: {self.use_cuda}")
        self.logger.info(f"Stopword removal: {self.remove_stopwords}")

    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type based on model name."""
        model_lower = model_name.lower()

        if any(x in model_lower for x in ['sentence-transformers', 'bge', 'e5', 'gte']):
            return 'sentence-transformer'
        elif any(x in model_lower for x in ['llama', 'qwen', 'mistral', 'deepseek']):
            return 'causal-lm'
        else:
            return 'auto'

    def _setup_device(self, device: Optional[str]) -> str:
        if device:
            return device

        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3)

            self.logger.info(
                f"Using single GPU cuda:{gpu_id} → {gpu_name} ({memory_gb:.1f}GB)"
            )
            return f"cuda:{gpu_id}"
        else:
            self.logger.info("CUDA not available, using CPU")
            return "cpu"

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        if not self.use_cuda:
            return

        torch.cuda.empty_cache()

        try:
            torch.backends.cuda.enable_flash_sdp(True)
            self.logger.info("✓ Flash Attention enabled")
        except:
            self.logger.info("Flash Attention not available")

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.logger.info("✓ CUDA optimizations applied")

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU memory."""
        if not self.use_cuda:
            return 8

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)

            if memory_gb >= 40:
                return 32
            elif memory_gb >= 20:
                return 16
            elif memory_gb >= 10:
                return 12
            else:
                return 8
        except:
            return 8

    def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication if needed."""
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
                self.logger.info("✓ HuggingFace authentication successful")
            except Exception as e:
                self.logger.warning(f"Authentication failed: {e}")

    def remove_stopwords_from_text(self, text: str) -> str:
        """
        Remove stopwords from text while preserving structure.

        Args:
            text: Input text

        Returns:
            Text with stopwords removed
        """
        if not self.remove_stopwords or not text or not self.stop_words:
            return text

        try:
            # Try using NLTK tokenizer
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split if NLTK has issues
            tokens = text.lower().split()

        # Remove stopwords while preserving important punctuation
        filtered_tokens = []
        for token in tokens:
            # Keep the token if it's not a stopword or if it's important punctuation
            if token not in self.stop_words or token in ['.', ',', ';', ':', '!', '?']:
                filtered_tokens.append(token)

        # Reconstruct the text
        filtered_text = ' '.join(filtered_tokens)

        # Clean up spacing around punctuation
        filtered_text = re.sub(r'\s+([.,;:!?])', r'\1', filtered_text)
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

        return filtered_text

    def combine_text_fields(self, title: str, abstract: str, authors: str) -> str:
        """Combine title, abstract, and authors based on embedding mode."""
        # Clean and prepare each field
        title = str(title) if title and not pd.isna(title) else ""
        abstract = str(abstract) if abstract and not pd.isna(abstract) else ""
        authors = str(authors) if authors and not pd.isna(authors) else ""

        # Apply stopword removal if enabled
        if self.remove_stopwords:
            title = self.remove_stopwords_from_text(title)
            abstract = self.remove_stopwords_from_text(abstract)
            authors = self.remove_stopwords_from_text(authors)

        if self.embedding_mode == "title":
            return title
        elif self.embedding_mode == "abstract":
            return abstract
        elif self.embedding_mode == "authors":
            return authors
        elif self.embedding_mode == "title_abstract":
            combined = f"{title}"
            if abstract:
                combined += f" {abstract}"
            return combined
        elif self.embedding_mode == "combined":
            # Combine all fields with separators
            combined = f"{title}"
            if abstract:
                combined += f" [ABSTRACT] {abstract}"
            if authors:
                combined += f" [AUTHORS] {authors}"
            return combined
        else:
            # Default to title if unknown mode
            self.logger.warning(f"Unknown embedding mode '{self.embedding_mode}', using title")
            return title

    def load_model(self) -> None:
        """Load tokenizer and model with CUDA optimizations."""
        self.logger.info(f"Loading model: {self.model_name}")
        self._setup_huggingface_auth()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with CUDA optimizations
            model_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }

            if self.use_cuda:
                model_kwargs['torch_dtype'] = torch.float16
            else:
                model_kwargs['torch_dtype'] = torch.float32

            # Load model based on type
            if self.model_type == 'sentence-transformer':
                self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            elif self.model_type == 'causal-lm':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                if hasattr(self.model, 'base_model'):
                    self.model = self.model.base_model
            else:
                try:
                    self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
                    if hasattr(self.model, 'base_model'):
                        self.model = self.model.base_model

            # Enable gradient checkpointing for memory efficiency
            if self.gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            self.model.eval()

            # Only move to device if device_map wasn't used
            self.model = self.model.to(self.device)

            self.logger.info("✓ Model loaded successfully")

            # Log memory usage
            if self.use_cuda:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                cached = torch.cuda.memory_reserved() / (1024 ** 3)
                self.logger.info(f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Generate embeddings with CUDA optimizations."""
        if batch_size is None:
            batch_size = self.max_batch_size

        embeddings = []
        hidden_size = getattr(self.model.config, 'hidden_size', 768)

        # Enable autocast for mixed precision
        autocast_context = torch.no_grad()

        with autocast_context:
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating embeddings (batch={batch_size})"):
                batch_texts = texts[i:i + batch_size]

                # Filter out empty texts
                valid_texts = [t if t and not pd.isna(t) else "" for t in batch_texts]

                if all(not t for t in valid_texts):
                    # All texts are empty, return zero vectors
                    batch_embeddings = np.zeros((len(batch_texts), hidden_size))
                else:
                    # Tokenize batch with optimal settings
                    inputs = self.tokenizer(
                        valid_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_attention_mask=True,
                        return_token_type_ids=False
                    )

                    # Move to device efficiently
                    # When using device_map='auto', don't manually move inputs
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    # If using device_map, let the model handle device placement automatically

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                        # Get embeddings based on model type
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            embeddings_tensor = outputs.pooler_output
                        else:
                            # Optimized mean pooling
                            embeddings_tensor = outputs.last_hidden_state
                            attention_mask = inputs['attention_mask']

                            # Efficient mean pooling
                            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
                            sum_embeddings = torch.sum(embeddings_tensor * mask_expanded, dim=1)
                            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                            embeddings_tensor = sum_embeddings / sum_mask

                        # Convert to numpy efficiently
                        batch_embeddings = embeddings_tensor.detach().cpu().numpy()

                embeddings.append(batch_embeddings)

                # Clean up GPU memory periodically
                if self.use_cuda and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

        # Final cleanup
        if self.use_cuda:
            torch.cuda.empty_cache()
            gc.collect()

        return np.vstack(embeddings)

    def load_ogbn_arxiv_data(self, nodes_csv_path: str) -> Tuple[
        pd.DataFrame, List[str], List[int], List, List[int], List[str], List[str]]:
        """
        Load OGBN-ArXiv nodes data with title, abstract, and authors.

        Returns:
            Tuple of (dataframe, combined_texts, node_ids, mag_paper_ids, class_indices, abstracts, authors)
        """
        self.logger.info(f"Loading OGBN-ArXiv data from {nodes_csv_path}")

        # Load CSV with memory optimization
        df = pd.read_csv(nodes_csv_path, low_memory=False)

        # Validate expected columns
        expected_columns = ['node_id', 'mag_paper_id', 'title', 'class_idx']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        self.logger.info(f"Loaded {len(df)} papers from OGBN-ArXiv dataset")
        self.logger.info(f"Available columns: {list(df.columns)}")

        # Memory optimization: convert to appropriate dtypes
        df['node_id'] = df['node_id'].astype('int32')
        df['class_idx'] = df['class_idx'].astype('int16')
        df['mag_paper_id'] = pd.to_numeric(df['mag_paper_id'], errors='coerce')

        # Basic statistics
        self.logger.info(f"Number of unique classes: {df['class_idx'].nunique()}")
        self.logger.info(f"Class range: {df['class_idx'].min()} - {df['class_idx'].max()}")

        # Handle missing data
        missing_titles = df['title'].isna().sum()
        missing_abstracts = df['abstract'].isna().sum() if 'abstract' in df.columns else len(df)
        missing_authors = df['authors'].isna().sum() if 'authors' in df.columns else len(df)

        self.logger.info(
            f"Missing data - Titles: {missing_titles}, Abstracts: {missing_abstracts}, Authors: {missing_authors}")

        # Fill missing values
        df['title'] = df['title'].fillna("Missing title")
        df['abstract'] = df['abstract'].fillna("") if 'abstract' in df.columns else ""
        df['authors'] = df['authors'].fillna("") if 'authors' in df.columns else ""

        # Combine text fields based on embedding mode
        self.logger.info(f"Combining text fields using mode: {self.embedding_mode}")
        if self.remove_stopwords:
            self.logger.info("Applying stopword removal preprocessing...")

        combined_texts = []
        for _, row in df.iterrows():
            combined_text = self.combine_text_fields(
                title=row['title'],
                abstract=row.get('abstract', ''),
                authors=row.get('authors', '')
            )
            combined_texts.append(combined_text)

        # Extract data as lists for memory efficiency
        node_ids = df['node_id'].tolist()
        mag_paper_ids = df['mag_paper_id'].tolist()
        class_indices = df['class_idx'].tolist()
        abstracts = df['abstract'].tolist() if 'abstract' in df.columns else [''] * len(df)
        authors = df['authors'].tolist() if 'authors' in df.columns else [''] * len(df)

        # Log text statistics
        avg_text_length = np.mean([len(text) for text in combined_texts])
        if self.remove_stopwords:
            self.logger.info(f"Average text length after stopword removal: {avg_text_length:.0f} characters")
        else:
            self.logger.info(f"Average combined text length: {avg_text_length:.0f} characters")

        return df, combined_texts, node_ids, mag_paper_ids, class_indices, abstracts, authors

    def create_embeddings_with_labels(self, embeddings: np.ndarray, class_indices: List[int],
                                      mag_paper_ids: List, node_ids: List[int] = None,
                                      use_mag_ids: bool = True) -> np.ndarray:
        """Combine embeddings with class labels and paper identifiers."""
        # Convert to numpy arrays with memory optimization
        class_indices_array = np.array(class_indices, dtype=np.int16).reshape(-1, 1)

        if use_mag_ids:
            # Use MAG paper IDs as primary identifier
            processed_mag_ids = []
            for mag_id in mag_paper_ids:
                if mag_id is None or pd.isna(mag_id):
                    processed_mag_ids.append(-1)
                else:
                    processed_mag_ids.append(int(mag_id))

            paper_ids_array = np.array(processed_mag_ids, dtype=np.int64).reshape(-1, 1)
            identifier_type = "mag_paper_id"
        else:
            if node_ids is None:
                raise ValueError("node_ids must be provided when use_mag_ids=False")
            paper_ids_array = np.array(node_ids, dtype=np.int32).reshape(-1, 1)
            identifier_type = "node_id"

        # Combine embeddings with labels: [embeddings, paper_id, class_idx]
        embeddings_with_labels = np.hstack([
            embeddings.astype(np.float32),
            paper_ids_array.astype(np.float32),
            class_indices_array.astype(np.float32)
        ])

        self.logger.info(f"Created embeddings with labels, shape: {embeddings_with_labels.shape}")
        self.logger.info(f"Format: [embedding_dims({embeddings.shape[1]}), {identifier_type}, class_idx]")

        return embeddings_with_labels

    def process_ogbn_arxiv_data(self,
                                nodes_csv: str = "../data/processed/ogbn_arxiv_nodes.csv",
                                output_dir: str = "../embeddings",
                                batch_size: int = None,
                                include_class_labels: bool = True,
                                use_mag_ids: bool = True,
                                save_intermediate: bool = True) -> Dict[str, Any]:
        """Process OGBN-ArXiv citation network data with CUDA optimizations."""

        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self.max_batch_size

        # Load OGBN-ArXiv data
        df, combined_texts, node_ids, mag_paper_ids, class_indices, abstracts, authors = self.load_ogbn_arxiv_data(
            nodes_csv)

        self.logger.info(f"Processing {len(combined_texts)} OGBN-ArXiv papers")
        self.logger.info(f"Using batch size: {batch_size}")
        self.logger.info(f"Using {'MAG paper IDs' if use_mag_ids else 'node indices'} as identifiers")
        self.logger.info(f"Embedding mode: {self.embedding_mode}")
        self.logger.info(f"Stopword removal: {self.remove_stopwords}")

        # Generate embeddings for combined texts
        embeddings = self.generate_embeddings_batch(combined_texts, batch_size)

        # Create embeddings with class labels if requested
        if include_class_labels:
            embeddings_with_labels = self.create_embeddings_with_labels(
                embeddings, class_indices, mag_paper_ids, node_ids, use_mag_ids
            )
        else:
            embeddings_with_labels = embeddings

        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create model-specific filename with embedding mode and stopword indicator
        model_short_name = self.model_name.split('/')[-1].lower().replace('-', '_')
        stopword_suffix = "_no_stopwords" if self.remove_stopwords else ""

        # Save embeddings as numpy array
        if include_class_labels:
            id_suffix = "mag_ids" if use_mag_ids else "node_ids"
            embeddings_file = output_path / f"{model_short_name}_{self.embedding_mode}_embeddings_with_labels_{id_suffix}{stopword_suffix}.npy"
        else:
            embeddings_file = output_path / f"{model_short_name}_{self.embedding_mode}_embeddings{stopword_suffix}.npy"

        # Save with memory optimization
        np.save(embeddings_file, embeddings_with_labels.astype(np.float32))
        self.logger.info(f"Saved embeddings to {embeddings_file}")

        # Save metadata
        metadata = {
            'model': self.model_name,
            'embedding_mode': self.embedding_mode,
            'stopword_removal': self.remove_stopwords,
            'stopwords_language': self.stopwords_lang if self.remove_stopwords else None,
            'num_papers': len(node_ids),
            'embedding_dim': embeddings.shape[1],
            'num_classes': len(set(class_indices)),
            'class_range': f"{min(class_indices)}-{max(class_indices)}",
            'device': self.device,
            'batch_size': batch_size,
            'include_labels': include_class_labels,
            'use_mag_ids': use_mag_ids,
            'identifier_type': 'mag_paper_id' if use_mag_ids else 'node_id',
            'output_shape': embeddings_with_labels.shape,
            'missing_mag_ids': sum(1 for x in mag_paper_ids if x is None or pd.isna(x)),
            'cuda_optimized': self.use_cuda,
            'mixed_precision': self.mixed_precision,
            'avg_text_length': np.mean([len(text) for text in combined_texts])
        }

        # Print summary
        self._print_summary(metadata, embeddings_with_labels, output_path, include_class_labels, use_mag_ids)

        return {
            'embeddings_file': str(embeddings_file),
            'metadata': metadata,
            'shape': embeddings_with_labels.shape,
            'embedding_dim': embeddings.shape[1],
            'num_classes': len(set(class_indices)),
            'identifier_type': 'mag_paper_id' if use_mag_ids else 'node_id',
            'embedding_mode': self.embedding_mode,
            'stopword_removal': self.remove_stopwords
        }

    def _print_summary(self, metadata, embeddings_with_labels, output_path, include_class_labels, use_mag_ids):
        """Print generation summary."""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("CUDA-OPTIMIZED EMBEDDING GENERATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {metadata['model']}")
        self.logger.info(f"Embedding mode: {metadata['embedding_mode']}")
        self.logger.info(f"Stopword removal: {metadata['stopword_removal']}")
        if metadata['stopword_removal']:
            self.logger.info(f"Stopwords language: {metadata['stopwords_language']}")
        self.logger.info(f"Device: {metadata['device']}")
        self.logger.info(f"CUDA optimizations: {metadata['cuda_optimized']}")
        self.logger.info(f"Batch size: {metadata['batch_size']}")
        self.logger.info(f"Papers processed: {metadata['num_papers']}")
        self.logger.info(f"Embedding dimension: {metadata['embedding_dim']}")
        self.logger.info(f"Number of classes: {metadata['num_classes']}")
        self.logger.info(f"Average text length: {metadata['avg_text_length']:.0f} chars")
        self.logger.info(f"Identifier type: {'MAG paper ID' if use_mag_ids else 'Node ID'}")

        if include_class_labels:
            identifier_name = "mag_paper_id" if use_mag_ids else "node_id"
            self.logger.info(
                f"Output shape: {embeddings_with_labels.shape} [embeddings + {identifier_name} + class_idx]")
        else:
            self.logger.info(f"Output shape: {embeddings_with_labels.shape} [embeddings only]")

        self.logger.info(f"Files saved to: {output_path}")

        # GPU memory info
        if metadata['cuda_optimized']:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            cached = torch.cuda.memory_reserved() / (1024 ** 3)
            self.logger.info(f"Final GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate CUDA-optimized embeddings for OGBN-ArXiv")
    parser.add_argument("--model", type=str, default="llama-3.2-3b", help="Model name")
    parser.add_argument("--embedding-mode", type=str, default="title",
                        choices=["title", "abstract", "authors", "title_abstract", "combined"],
                        help="What to embed: title, abstract, authors, title_abstract, or combined")

    # Stopword removal options
    parser.add_argument("--remove-stopwords", action="store_true",
                        help="Remove stopwords before generating embeddings")
    parser.add_argument("--stopwords-lang", type=str, default="english",
                        help="Language for stopwords (default: english)")

    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto-determined if None)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--nodes-csv", type=str, default="../data/processed/ogbn_arxiv_nodes.csv",
                        help="Path to nodes CSV")
    parser.add_argument("--output-dir", type=str, default="../embeddings", help="Output directory")
    parser.add_argument("--no-labels", action="store_true", help="Don't include class labels")
    parser.add_argument("--use-node-ids", action="store_true", help="Use node IDs instead of MAG paper IDs")
    parser.add_argument("--no-intermediate", action="store_true", help="Don't save intermediate CSV files")

    args = parser.parse_args()

    # Initialize CUDA-optimized generator with stopword removal option
    generator = CUDAOptimizedEmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        embedding_mode=args.embedding_mode,
        remove_stopwords=args.remove_stopwords,
        stopwords_lang=args.stopwords_lang
    )

    # Load model
    generator.load_model()

    # Process data
    result = generator.process_ogbn_arxiv_data(
        nodes_csv=args.nodes_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        include_class_labels=not args.no_labels,
        use_mag_ids=not args.use_node_ids,
        save_intermediate=not args.no_intermediate
    )

    print("\n" + "=" * 60)
    print("CUDA-OPTIMIZED EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Model: {result['metadata']['model']}")
    print(f"Embedding mode: {result['embedding_mode']}")
    print(f"Stopword removal: {'Yes' if result['stopword_removal'] else 'No'}")
    print(f"Shape: {result['shape']}")
    print(f"Saved to: {result['embeddings_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()