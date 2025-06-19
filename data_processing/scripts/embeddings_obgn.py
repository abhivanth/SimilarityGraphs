import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    # LLaMA models
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-3b": "meta-llama/Llama-3.1-3B",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B",
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",

    # Qwen models
    "qwen-3.3-32b": "Qwen/Qwen2.5-32B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",

    # DeepSeek models
    "deepseek-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


class EmbeddingGenerator:
    """Generate embeddings for citation network papers using various language models."""

    def __init__(self, model_name: str = "llama-3.2-3b", device: Optional[str] = None):
        """
        Initialize embedding generator.

        Args:
            model_name: Model identifier (key from MODEL_CONFIGS or HuggingFace model ID)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Setup logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Resolve model name
        self.model_name = MODEL_CONFIGS.get(model_name, model_name)
        self.model_type = self._detect_model_type(self.model_name)

        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None

        self.logger.info(f"Initialized with model: {self.model_name}")
        self.logger.info(f"Model type: {self.model_type}")

    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type based on model name."""
        model_lower = model_name.lower()

        if any(x in model_lower for x in ['sentence-transformers', 'bge', 'e5', 'gte']):
            return 'sentence-transformer'
        elif any(x in model_lower for x in ['llama', 'qwen', 'mistral', 'deepseek']):
            return 'causal-lm'
        else:
            return 'auto'

    def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication if needed."""
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

        if hf_token:
            try:
                login(token=hf_token, add_to_git_credential=False)
                self.logger.info("✓ HuggingFace authentication successful")
            except Exception as e:
                self.logger.warning(f"Authentication failed: {e}")

    def _setup_device(self, device: Optional[str]) -> str:
        """Setup computation device."""
        if device:
            return device

        if torch.cuda.is_available():
            self.logger.info("CUDA available, using GPU")
            return "cuda"
        else:
            self.logger.info("CUDA not available, using CPU")
            return "cpu"

    def load_model(self) -> None:
        """Load tokenizer and model based on model type."""
        self.logger.info(f"Loading model: {self.model_name}")
        self._setup_huggingface_auth()

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            # Load model based on type
            if self.model_type == 'sentence-transformer':
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
            elif self.model_type == 'causal-lm':
                # For causal LMs, we'll use the base model for embeddings
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                ).base_model
            else:
                # Auto-detect
                try:
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        trust_remote_code=True
                    )
                except:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=dtype,
                        trust_remote_code=True
                    ).base_model

            self.model.eval()
            self.model.to(self.device)

            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings_batch(self, texts: list, batch_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Array of embeddings
        """
        embeddings = []
        hidden_size = getattr(self.model.config, 'hidden_size', 768)

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Filter out empty texts
            valid_texts = [t if t and not pd.isna(t) else "" for t in batch_texts]

            if all(not t for t in valid_texts):
                # All texts are empty, return zero vectors
                batch_embeddings = np.zeros((len(batch_texts), hidden_size))
            else:
                # Tokenize batch
                inputs = self.tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Get embeddings based on model type
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        batch_embeddings = outputs.pooler_output.cpu().numpy()
                    else:
                        # Mean pooling
                        embeddings_tensor = outputs.last_hidden_state
                        attention_mask = inputs['attention_mask']

                        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
                        sum_embeddings = torch.sum(embeddings_tensor * mask_expanded, dim=1)
                        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def load_ogbn_arxiv_data(self, nodes_csv_path: str) -> Tuple[pd.DataFrame, list, list, list, list]:
        """
        Load OGBN-ArXiv nodes data with validation.

        Args:
            nodes_csv_path: Path to the ogbn_arxiv_nodes.csv file

        Returns:
            Tuple of (dataframe, titles, node_ids, mag_paper_ids, class_indices)
        """
        self.logger.info(f"Loading OGBN-ArXiv data from {nodes_csv_path}")

        # Load the CSV
        df = pd.read_csv(nodes_csv_path)

        # Validate expected columns
        expected_columns = ['node_id', 'mag_paper_id', 'title', 'class_idx']
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        self.logger.info(f"Loaded {len(df)} papers from OGBN-ArXiv dataset")
        self.logger.info(f"Columns: {df.columns.tolist()}")

        # Basic statistics
        self.logger.info(f"Number of unique classes: {df['class_idx'].nunique()}")
        self.logger.info(f"Class range: {df['class_idx'].min()} - {df['class_idx'].max()}")

        # Check MAG paper ID availability
        missing_mag_ids = df['mag_paper_id'].isna().sum()
        if missing_mag_ids > 0:
            self.logger.warning(f"Found {missing_mag_ids} papers with missing MAG paper IDs")

        # Handle missing titles
        missing_titles = df['title'].isna().sum()
        if missing_titles > 0:
            self.logger.warning(f"Found {missing_titles} papers with missing titles")
            df['title'] = df['title'].fillna(f"Missing title for paper")

        # Extract data
        titles = df['title'].tolist()
        node_ids = df['node_id'].tolist()
        mag_paper_ids = df['mag_paper_id'].tolist()
        class_indices = df['class_idx'].tolist()

        return df, titles, node_ids, mag_paper_ids, class_indices

    def create_embeddings_with_labels(self,
                                      embeddings: np.ndarray,
                                      class_indices: list,
                                      mag_paper_ids: list,
                                      node_ids: list = None,
                                      use_mag_ids: bool = True) -> np.ndarray:
        """
        Combine embeddings with class labels and paper identifiers.

        Args:
            embeddings: Embedding vectors [n_samples, embedding_dim]
            class_indices: Class labels [n_samples]
            mag_paper_ids: MAG paper IDs [n_samples]
            node_ids: Node IDs [n_samples] (optional)
            use_mag_ids: Whether to use MAG IDs or node IDs as primary identifier

        Returns:
            Combined array [n_samples, embedding_dim + 2] where last two columns are [paper_id, class_idx]
        """
        # Convert to numpy arrays
        class_indices_array = np.array(class_indices, dtype=np.float32).reshape(-1, 1)

        if use_mag_ids:
            # Use MAG paper IDs as primary identifier
            # Handle None values in MAG paper IDs
            processed_mag_ids = []
            for mag_id in mag_paper_ids:
                if mag_id is None or pd.isna(mag_id):
                    processed_mag_ids.append(-1.0)  # Use -1 for missing MAG IDs
                else:
                    processed_mag_ids.append(float(mag_id))

            paper_ids_array = np.array(processed_mag_ids, dtype=np.float32).reshape(-1, 1)
            identifier_type = "mag_paper_id"
        else:
            # Use node IDs as primary identifier
            if node_ids is None:
                raise ValueError("node_ids must be provided when use_mag_ids=False")
            paper_ids_array = np.array(node_ids, dtype=np.float32).reshape(-1, 1)
            identifier_type = "node_id"

        # Combine embeddings with labels: [embeddings, paper_id, class_idx]
        embeddings_with_labels = np.hstack([
            embeddings.astype(np.float32),
            paper_ids_array,
            class_indices_array
        ])

        self.logger.info(f"Created embeddings with labels, shape: {embeddings_with_labels.shape}")
        self.logger.info(f"Format: [embedding_dims({embeddings.shape[1]}), {identifier_type}, class_idx]")

        return embeddings_with_labels

    def process_ogbn_arxiv_data(self,
                                nodes_csv: str = "../data/processed/ogbn_arxiv_nodes.csv",
                                output_dir: str = "../embeddings",
                                batch_size: int = 8,
                                include_class_labels: bool = True,
                                use_mag_ids: bool = True) -> Dict[str, Any]:
        """
        Process OGBN-ArXiv citation network data and generate embeddings with class labels.

        Args:
            nodes_csv: Path to ogbn_arxiv_nodes.csv file
            output_dir: Directory to save embeddings
            batch_size: Batch size for processing
            include_class_labels: Whether to include class labels in output
            use_mag_ids: Whether to use MAG paper IDs (True) or node IDs (False) as identifiers

        Returns:
            Dictionary with embedding info
        """
        # Load OGBN-ArXiv data
        df, titles, node_ids, mag_paper_ids, class_indices = self.load_ogbn_arxiv_data(nodes_csv)

        self.logger.info(f"Processing {len(titles)} OGBN-ArXiv papers")
        self.logger.info(f"Using {'MAG paper IDs' if use_mag_ids else 'node indices'} as identifiers")

        # Generate embeddings for titles
        embeddings = self.generate_embeddings_batch(titles, batch_size)

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

        # Create model-specific filename
        model_short_name = self.model_name.split('/')[-1].lower().replace('-', '_')

        # Save embeddings as numpy array
        if include_class_labels:
            id_suffix = "mag_ids" if use_mag_ids else "node_ids"
            embeddings_file = output_path / f"{model_short_name}_embeddings_with_labels_{id_suffix}.npy"
        else:
            embeddings_file = output_path / f"{model_short_name}_embeddings.npy"

        np.save(embeddings_file, embeddings_with_labels)
        self.logger.info(f"Saved embeddings to {embeddings_file}")

        # Save detailed CSV with all information
        if include_class_labels:
            identifier_col = 'mag_paper_id' if use_mag_ids else 'node_id'
            identifier_values = mag_paper_ids if use_mag_ids else node_ids

            embeddings_df = pd.DataFrame({
                'node_id': node_ids,
                'mag_paper_id': mag_paper_ids,
                'class_idx': class_indices,
                'title': titles,
                **{f'dim_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
            })
            csv_file = output_path / f"{model_short_name}_embeddings_with_metadata_{identifier_col}.csv"
        else:
            embeddings_df = pd.DataFrame({
                'node_id': node_ids,
                'mag_paper_id': mag_paper_ids,
                **{f'dim_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
            })
            csv_file = output_path / f"{model_short_name}_embeddings.csv"

        embeddings_df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved detailed CSV to {csv_file}")

        # Save class mapping for reference
        class_mapping_file = output_path / f"{model_short_name}_class_distribution.csv"
        class_dist = pd.DataFrame({
            'class_idx': class_indices
        }).value_counts().reset_index()
        class_dist.columns = ['class_idx', 'count']
        class_dist = class_dist.sort_values('class_idx')
        class_dist.to_csv(class_mapping_file, index=False)

        # Save metadata
        metadata = {
            'model': self.model_name,
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
            'missing_mag_ids': sum(1 for x in mag_paper_ids if x is None or pd.isna(x))
        }

        # Print summary statistics
        self.logger.info("\n" + "=" * 50)
        self.logger.info("EMBEDDING GENERATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Papers processed: {len(node_ids)}")
        self.logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        self.logger.info(f"Number of classes: {len(set(class_indices))}")
        self.logger.info(f"Identifier type: {'MAG paper ID' if use_mag_ids else 'Node ID'}")
        if include_class_labels:
            identifier_name = "mag_paper_id" if use_mag_ids else "node_id"
            self.logger.info(
                f"Output shape: {embeddings_with_labels.shape} [embeddings + {identifier_name} + class_idx]")
        else:
            self.logger.info(f"Output shape: {embeddings_with_labels.shape} [embeddings only]")
        self.logger.info(f"Files saved to: {output_path}")

        return {
            'embeddings_file': str(embeddings_file),
            'csv_file': str(csv_file),
            'class_mapping_file': str(class_mapping_file),
            'metadata': metadata,
            'shape': embeddings_with_labels.shape,
            'embedding_dim': embeddings.shape[1],
            'num_classes': len(set(class_indices)),
            'identifier_type': 'mag_paper_id' if use_mag_ids else 'node_id'
        }

    def process_citation_data(self,
                              nodes_csv: str = "../data/processed/nodes.csv",
                              output_dir: str = "../embeddings",
                              batch_size: int = 8) -> Dict[str, Any]:
        """
        Process citation network data and generate embeddings.
        (Original method maintained for backward compatibility)

        Args:
            nodes_csv: Path to nodes CSV file
            output_dir: Directory to save embeddings
            batch_size: Batch size for processing

        Returns:
            Dictionary with embedding info
        """
        # Load data
        self.logger.info(f"Loading data from {nodes_csv}")
        df = pd.read_csv(nodes_csv)

        # Extract titles
        titles = df['title'].tolist()
        paper_ids = df['paper_id'].tolist()

        self.logger.info(f"Processing {len(titles)} papers")

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(titles, batch_size)

        # Prepare output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save embeddings as numpy array
        model_short_name = self.model_name.split('/')[-1].lower().replace('-', '_')
        embeddings_file = output_path / f"{model_short_name}_embeddings.npy"
        np.save(embeddings_file, embeddings)
        self.logger.info(f"Saved embeddings to {embeddings_file}")

        # Save embeddings with paper IDs for reference
        embeddings_with_ids = pd.DataFrame({
            'paper_id': paper_ids,
            **{f'dim_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
        })

        csv_file = output_path / f"{model_short_name}_embeddings.csv"
        embeddings_with_ids.to_csv(csv_file, index=False)
        self.logger.info(f"Saved embeddings CSV to {csv_file}")

        # Save metadata
        metadata = {
            'model': self.model_name,
            'num_papers': len(paper_ids),
            'embedding_dim': embeddings.shape[1],
            'device': self.device,
            'batch_size': batch_size
        }

        return {
            'embeddings_file': str(embeddings_file),
            'csv_file': str(csv_file),
            'metadata': metadata,
            'shape': embeddings.shape
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for OGBN-ArXiv citation network")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.2-3b",
        help="Model name (key from MODEL_CONFIGS or HuggingFace model ID)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--nodes-csv",
        type=str,
        default="../data/processed/ogbn_arxiv_nodes.csv",
        help="Path to OGBN-ArXiv nodes CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't include class labels in output"
    )
    parser.add_argument(
        "--use-node-ids",
        action="store_true",
        help="Use node IDs instead of MAG paper IDs as identifiers"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = EmbeddingGenerator(
        model_name=args.model,
        device=args.device
    )

    # Load model
    generator.load_model()

    # Process OGBN-ArXiv data
    result = generator.process_ogbn_arxiv_data(
        nodes_csv=args.nodes_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        include_class_labels=not args.no_labels,
        use_mag_ids=not args.use_node_ids  # Default to MAG IDs, flip if --use-node-ids
    )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Model: {result['metadata']['model']}")
    print(f"Shape: {result['shape']}")
    print(f"Embedding dimensions: {result['embedding_dim']}")
    print(f"Number of classes: {result['num_classes']}")
    print(f"Saved to: {result['embeddings_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()