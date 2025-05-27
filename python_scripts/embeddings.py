import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate embeddings for citation network papers using LLaMA models."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", device: Optional[str] = None):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
        """Load tokenizer and model."""
        self.logger.info(f"Loading model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with appropriate dtype
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True
            )

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

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Filter out empty texts
            valid_texts = [t if t and not pd.isna(t) else "" for t in batch_texts]

            if all(not t for t in valid_texts):
                # All texts are empty, return zero vectors
                batch_embeddings = np.zeros((len(batch_texts), self.model.config.hidden_size))
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

                    # Mean pooling
                    embeddings_tensor = outputs.last_hidden_state
                    attention_mask = inputs['attention_mask']

                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_tensor.size()).float()
                    sum_embeddings = torch.sum(embeddings_tensor * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def process_citation_data(self, nodes_csv: str, model_name: str, output_dir: str = "embeddings",
                              batch_size: int = 8) -> Dict[str, Any]:
        """
        Process citation network data and generate embeddings.

        Args:
            nodes_csv: Path to nodes CSV file
            output_dir: Directory to save embeddings
            batch_size: Batch size for processing

        Returns:
            Dictionary with embedding info
            :param nodes_csv:
            :param output_dir:
            :param batch_size:
            :param model_name:
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
        embeddings_file = output_path / "llama_3_2_3b_embeddings.npy"
        np.save(embeddings_file, embeddings)
        self.logger.info(f"Saved embeddings to {embeddings_file}")

        # Save embeddings with paper IDs for reference
        embeddings_with_ids = pd.DataFrame({
            'paper_id': paper_ids,
            **{f'dim_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
        })

        short_model_name = model_name.rsplit("/", 1)[-1]
        csv_file = output_path / f"{short_model_name}_embeddings.csv"
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
    """Generate embeddings for citation network using LLaMA 3.2 3B."""
    # Initialize generator
    model_name = "meta-llama/Llama-3.2-3B"  # meta-llama/Llama-3.2-3B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    generator = EmbeddingGenerator(
        model_name=model_name,
        device=None  # Auto-detect
    )

    # Load model
    generator.load_model()

    # Process citation data
    result = generator.process_citation_data(
        nodes_csv="data/processed/nodes.csv",
        output_dir="embeddings",
        batch_size=8,  # Adjust based on GPU memory
        model_name=model_name
    )

    print("\nEmbedding generation complete!")
    print(f"Shape: {result['shape']}")
    print(f"Saved to: {result['embeddings_file']}")


if __name__ == "__main__":
    main()
