import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging
from tqdm import tqdm
import os
import gc
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class Specter2BaselineGenerator:
    """Specter2 baseline embedding generator for citation network papers."""

    def __init__(self,
                 device: Optional[str] = None,
                 embedding_mode: str = "title_abstract",
                 remove_stopwords: bool = False,
                 stopwords_lang: str = "english"):
        """
        Initialize Specter2 baseline embedding generator.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            embedding_mode: What to embed ('title', 'abstract', 'title_abstract')
            remove_stopwords: Whether to remove stopwords before embedding
            stopwords_lang: Language for stopwords (default: 'english')
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Specter2 model - fixed for baseline
        self.model_name = "allenai/specter2_base"
        self.embedding_mode = embedding_mode
        self.remove_stopwords = remove_stopwords
        self.stopwords_lang = stopwords_lang

        # Setup device
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None

        # CUDA optimization settings
        self.use_cuda = self.device == "cuda"
        self.mixed_precision = self.use_cuda

        # Batch size optimization
        self.max_batch_size = self._get_optimal_batch_size()

        # Initialize CUDA optimizations if available
        if self.use_cuda:
            self._setup_cuda_optimizations()

        # Initialize stopwords if needed
        if self.remove_stopwords:
            self.stop_words = set(stopwords.words(self.stopwords_lang))
            self.logger.info(f"Initialized stopword removal for language: {self.stopwords_lang}")
            self.logger.info(f"Number of stopwords: {len(self.stop_words)}")
        else:
            self.stop_words = None

        self.logger.info(f"Initialized Specter2 Baseline Generator")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Embedding mode: {self.embedding_mode}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Stopword removal: {self.remove_stopwords}")

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device."""
        if device:
            return device

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            self.logger.info(f"CUDA available: {gpu_name} ({memory_gb:.1f}GB)")
            return "cuda"
        else:
            self.logger.info("CUDA not available, using CPU")
            return "cpu"

    def _setup_cuda_optimizations(self):
        """Setup CUDA-specific optimizations."""
        if not self.use_cuda:
            return

        torch.cuda.empty_cache()

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.logger.info("✓ CUDA optimizations applied")

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on available GPU memory."""
        if not self.use_cuda:
            return 16  # CPU batch size

        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024 ** 3)

            # Specter2 is relatively lightweight, so we can use larger batches
            if memory_gb >= 40:
                return 64
            elif memory_gb >= 20:
                return 32
            elif memory_gb >= 10:
                return 24
            else:
                return 16
        except:
            return 16

    def remove_stopwords_from_text(self, text: str) -> str:
        """
        Remove stopwords from text while preserving structure.

        Args:
            text: Input text

        Returns:
            Text with stopwords removed
        """
        if not self.remove_stopwords or not text:
            return text

        # Tokenize the text
        tokens = word_tokenize(text.lower())

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

    def combine_text_fields(self, title: str, abstract: str) -> str:
        """
        Combine title and abstract for Specter2 input.

        Args:
            title: Paper title
            abstract: Paper abstract

        Returns:
            Combined text formatted for Specter2
        """
        # Clean and prepare each field
        title = str(title) if title and not pd.isna(title) else ""
        abstract = str(abstract) if abstract and not pd.isna(abstract) else ""

        # Apply stopword removal if enabled
        if self.remove_stopwords:
            title = self.remove_stopwords_from_text(title)
            abstract = self.remove_stopwords_from_text(abstract)

        # Combine based on embedding mode
        if self.embedding_mode == "title":
            return title
        elif self.embedding_mode == "abstract":
            return abstract
        elif self.embedding_mode == "title_abstract":
            # Specter2 expects title and abstract separated
            if title and abstract:
                # Use [SEP] token or just concatenate with space
                return f"{title} [SEP] {abstract}"
            elif title:
                return title
            else:
                return abstract
        else:
            # Default to title_abstract
            return f"{title} [SEP] {abstract}" if title and abstract else title or abstract

    def load_model(self) -> None:
        """Load Specter2 model and tokenizer."""
        self.logger.info(f"Loading Specter2 model: {self.model_name}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load model with optimizations
            model_kwargs = {
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }

            if self.use_cuda:
                model_kwargs['torch_dtype'] = torch.float16
            else:
                model_kwargs['torch_dtype'] = torch.float32

            # Load Specter2 model
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self.model.eval()

            # Move to device
            self.model = self.model.to(self.device)

            self.logger.info("✓ Specter2 model loaded successfully")

            # Log memory usage if CUDA
            if self.use_cuda:
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                cached = torch.cuda.memory_reserved() / (1024 ** 3)
                self.logger.info(f"GPU memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")

        except Exception as e:
            self.logger.error(f"Failed to load Specter2 model: {e}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Generate Specter2 embeddings for a batch of texts.

        Args:
            texts: List of paper texts (title + abstract)
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings
        """
        if batch_size is None:
            batch_size = self.max_batch_size

        embeddings = []

        # Process in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size),
                          desc=f"Generating Specter2 embeddings (batch={batch_size})"):
                batch_texts = texts[i:i + batch_size]

                # Filter out empty texts
                valid_texts = [t if t and not pd.isna(t) else "" for t in batch_texts]

                if all(not t for t in valid_texts):
                    # All texts are empty, return zero vectors
                    batch_embeddings = np.zeros((len(batch_texts), 768))  # Specter2 dimension
                else:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        valid_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_attention_mask=True
                    )

                    # Move to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Generate embeddings
                    outputs = self.model(**inputs)

                    # Specter2 uses CLS token embedding
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embeddings_tensor = outputs.pooler_output
                    else:
                        # Use CLS token (first token)
                        embeddings_tensor = outputs.last_hidden_state[:, 0, :]

                    # Convert to numpy
                    batch_embeddings = embeddings_tensor.cpu().numpy()

                embeddings.append(batch_embeddings)

                # Clean GPU memory periodically
                if self.use_cuda and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

        # Final cleanup
        if self.use_cuda:
            torch.cuda.empty_cache()
            gc.collect()

        return np.vstack(embeddings)

    def load_ogbn_arxiv_data(self, nodes_csv_path: str) -> Tuple[pd.DataFrame, List[str], List[int], List, List[int]]:
        """
        Load OGBN-ArXiv nodes data.

        Returns:
            Tuple of (dataframe, combined_texts, node_ids, mag_paper_ids, class_indices)
        """
        self.logger.info(f"Loading OGBN-ArXiv data from {nodes_csv_path}")

        # Load CSV
        df = pd.read_csv(nodes_csv_path, low_memory=False)

        # Validate expected columns
        expected_columns = ['node_id', 'mag_paper_id', 'title', 'class_idx']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        self.logger.info(f"Loaded {len(df)} papers from OGBN-ArXiv dataset")

        # Convert to appropriate dtypes for memory efficiency
        df['node_id'] = df['node_id'].astype('int32')
        df['class_idx'] = df['class_idx'].astype('int16')
        df['mag_paper_id'] = pd.to_numeric(df['mag_paper_id'], errors='coerce')

        # Log statistics
        self.logger.info(f"Number of unique classes: {df['class_idx'].nunique()}")
        missing_titles = df['title'].isna().sum()
        missing_abstracts = df['abstract'].isna().sum() if 'abstract' in df.columns else len(df)
        self.logger.info(f"Missing data - Titles: {missing_titles}, Abstracts: {missing_abstracts}")

        # Fill missing values
        df['title'] = df['title'].fillna("")
        df['abstract'] = df['abstract'].fillna("") if 'abstract' in df.columns else ""

        # Combine text fields for Specter2
        self.logger.info(f"Combining text fields using mode: {self.embedding_mode}")
        if self.remove_stopwords:
            self.logger.info("Applying stopword removal preprocessing...")

        combined_texts = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing texts"):
            combined_text = self.combine_text_fields(
                title=row['title'],
                abstract=row.get('abstract', '')
            )
            combined_texts.append(combined_text)

        # Extract data as lists
        node_ids = df['node_id'].tolist()
        mag_paper_ids = df['mag_paper_id'].tolist()
        class_indices = df['class_idx'].tolist()

        # Log text statistics
        text_lengths = [len(text) for text in combined_texts]
        avg_length = np.mean(text_lengths)

        if self.remove_stopwords:
            self.logger.info(f"Average text length after stopword removal: {avg_length:.0f} characters")
        else:
            self.logger.info(f"Average text length: {avg_length:.0f} characters")

        return df, combined_texts, node_ids, mag_paper_ids, class_indices

    def create_embeddings_with_labels(self,
                                      embeddings: np.ndarray,
                                      class_indices: List[int],
                                      mag_paper_ids: List,
                                      node_ids: List[int] = None,
                                      use_mag_ids: bool = True) -> np.ndarray:
        """
        Combine embeddings with class labels and paper identifiers.

        Returns:
            Array with format: [embeddings, paper_id, class_idx]
        """
        # Convert to numpy arrays
        class_indices_array = np.array(class_indices, dtype=np.int16).reshape(-1, 1)

        if use_mag_ids:
            # Use MAG paper IDs
            processed_mag_ids = []
            for mag_id in mag_paper_ids:
                if mag_id is None or pd.isna(mag_id):
                    processed_mag_ids.append(-1)
                else:
                    processed_mag_ids.append(int(mag_id))

            paper_ids_array = np.array(processed_mag_ids, dtype=np.int64).reshape(-1, 1)
            identifier_type = "mag_paper_id"
        else:
            # Use node IDs
            if node_ids is None:
                raise ValueError("node_ids must be provided when use_mag_ids=False")
            paper_ids_array = np.array(node_ids, dtype=np.int32).reshape(-1, 1)
            identifier_type = "node_id"

        # Combine: [embeddings, paper_id, class_idx]
        embeddings_with_labels = np.hstack([
            embeddings.astype(np.float32),
            paper_ids_array.astype(np.float32),
            class_indices_array.astype(np.float32)
        ])

        self.logger.info(f"Created embeddings with labels, shape: {embeddings_with_labels.shape}")
        self.logger.info(f"Format: [embedding_dims(768), {identifier_type}, class_idx]")

        return embeddings_with_labels

    def process_ogbn_arxiv_data(self,
                                nodes_csv: str = "../data/processed/ogbn_arxiv_nodes.csv",
                                output_dir: str = "../embeddings",
                                batch_size: int = None,
                                include_class_labels: bool = True,
                                use_mag_ids: bool = True) -> Dict[str, Any]:
        """
        Process OGBN-ArXiv data and generate Specter2 embeddings.

        Returns:
            Dictionary with embedding file path and metadata
        """
        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self.max_batch_size

        # Load OGBN-ArXiv data
        df, combined_texts, node_ids, mag_paper_ids, class_indices = self.load_ogbn_arxiv_data(nodes_csv)

        self.logger.info(f"Processing {len(combined_texts)} papers with Specter2")
        self.logger.info(f"Using batch size: {batch_size}")
        self.logger.info(f"Stopword removal: {self.remove_stopwords}")

        # Generate Specter2 embeddings
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

        # Create filename with stopword removal indicator
        stopword_suffix = "_no_stopwords" if self.remove_stopwords else ""

        if include_class_labels:
            id_suffix = "mag_ids" if use_mag_ids else "node_ids"
            embeddings_file = output_path / f"specter2_baseline_{self.embedding_mode}_embeddings_with_labels_{id_suffix}{stopword_suffix}.npy"
        else:
            embeddings_file = output_path / f"specter2_baseline_{self.embedding_mode}_embeddings{stopword_suffix}.npy"

        # Save embeddings as .npy file
        np.save(embeddings_file, embeddings_with_labels.astype(np.float32))
        self.logger.info(f"✓ Saved embeddings to {embeddings_file}")

        # Create and save metadata
        metadata = {
            'model': 'allenai/specter2_base',
            'embedding_mode': self.embedding_mode,
            'stopword_removal': self.remove_stopwords,
            'stopwords_language': self.stopwords_lang if self.remove_stopwords else None,
            'num_papers': len(node_ids),
            'embedding_dim': 768,  # Specter2 dimension
            'num_classes': len(set(class_indices)),
            'class_range': f"{min(class_indices)}-{max(class_indices)}",
            'device': self.device,
            'batch_size': batch_size,
            'include_labels': include_class_labels,
            'use_mag_ids': use_mag_ids,
            'identifier_type': 'mag_paper_id' if use_mag_ids else 'node_id',
            'output_shape': embeddings_with_labels.shape,
            'missing_mag_ids': sum(1 for x in mag_paper_ids if x is None or pd.isna(x))
        }

        # Save metadata as JSON
        import json
        metadata_file = embeddings_file.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"✓ Saved metadata to {metadata_file}")

        # Print summary
        self._print_summary(metadata, embeddings_with_labels, output_path, include_class_labels, use_mag_ids)

        return {
            'embeddings_file': str(embeddings_file),
            'metadata': metadata,
            'shape': embeddings_with_labels.shape,
            'embedding_dim': 768,
            'num_classes': len(set(class_indices)),
            'identifier_type': 'mag_paper_id' if use_mag_ids else 'node_id',
            'embedding_mode': self.embedding_mode,
            'stopword_removal': self.remove_stopwords
        }

    def _print_summary(self, metadata, embeddings_with_labels, output_path, include_class_labels, use_mag_ids):
        """Print generation summary."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SPECTER2 BASELINE EMBEDDING GENERATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Model: Specter2 (allenai/specter2_base)")
        self.logger.info(f"Embedding mode: {metadata['embedding_mode']}")
        self.logger.info(f"Stopword removal: {metadata['stopword_removal']}")
        if metadata['stopword_removal']:
            self.logger.info(f"Stopwords language: {metadata['stopwords_language']}")
        self.logger.info(f"Device: {metadata['device']}")
        self.logger.info(f"Batch size: {metadata['batch_size']}")
        self.logger.info(f"Papers processed: {metadata['num_papers']}")
        self.logger.info(f"Embedding dimension: 768 (Specter2 fixed)")
        self.logger.info(f"Number of classes: {metadata['num_classes']}")
        self.logger.info(f"Identifier type: {'MAG paper ID' if use_mag_ids else 'Node ID'}")

        if include_class_labels:
            identifier_name = "mag_paper_id" if use_mag_ids else "node_id"
            self.logger.info(
                f"Output shape: {embeddings_with_labels.shape} [embeddings + {identifier_name} + class_idx]")
        else:
            self.logger.info(f"Output shape: {embeddings_with_labels.shape} [embeddings only]")

        self.logger.info(f"Files saved to: {output_path}")
        self.logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Specter2 baseline embeddings for OGBN-ArXiv")

    # Embedding options
    parser.add_argument("--embedding-mode", type=str, default="title_abstract",
                        choices=["title", "abstract", "title_abstract"],
                        help="What to embed: title, abstract, or title_abstract (default: title_abstract)")

    # Stopword removal options
    parser.add_argument("--remove-stopwords", action="store_true",
                        help="Remove stopwords before generating embeddings")
    parser.add_argument("--stopwords-lang", type=str, default="english",
                        help="Language for stopwords (default: english)")

    # Processing options
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for processing (auto-determined if not specified)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Device to use (auto-detected if not specified)")

    # Data options
    parser.add_argument("--nodes-csv", type=str, default="../data/processed/ogbn_arxiv_nodes.csv",
                        help="Path to OGBN-ArXiv nodes CSV file")
    parser.add_argument("--output-dir", type=str, default="../embeddings",
                        help="Output directory for embeddings")

    # Label options
    parser.add_argument("--no-labels", action="store_true",
                        help="Don't include class labels in output")
    parser.add_argument("--use-node-ids", action="store_true",
                        help="Use node IDs instead of MAG paper IDs")

    args = parser.parse_args()

    # Initialize Specter2 generator
    generator = Specter2BaselineGenerator(
        device=args.device,
        embedding_mode=args.embedding_mode,
        remove_stopwords=args.remove_stopwords,
        stopwords_lang=args.stopwords_lang
    )

    # Load Specter2 model
    generator.load_model()

    # Process data and generate embeddings
    result = generator.process_ogbn_arxiv_data(
        nodes_csv=args.nodes_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        include_class_labels=not args.no_labels,
        use_mag_ids=not args.use_node_ids
    )

    print("\n" + "=" * 70)
    print("SPECTER2 BASELINE EMBEDDING GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Model: Specter2 (Scientific Paper Embeddings)")
    print(f"Embedding mode: {result['embedding_mode']}")
    print(f"Stopword removal: {'Yes' if result['stopword_removal'] else 'No'}")
    print(f"Output shape: {result['shape']}")
    print(f"Embedding dimension: {result['embedding_dim']}")
    print(f"Number of classes: {result['num_classes']}")
    print(f"Saved to: {result['embeddings_file']}")
    print("=" * 70)


if __name__ == "__main__":
    main()