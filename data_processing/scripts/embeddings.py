import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
from typing import Optional, Dict, Any
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
    
    # DeepSeek models
    "deepseek-qwen-32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
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

    def process_citation_data(self, 
                            nodes_csv: str = "../data/processed/nodes.csv", 
                            output_dir: str = "../embeddings",
                            batch_size: int = 8) -> Dict[str, Any]:
        """
        Process citation network data and generate embeddings.

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
    
    parser = argparse.ArgumentParser(description="Generate embeddings for citation network")
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
        default="../data/processed/nodes.csv",
        help="Path to nodes CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="../embeddings",
        help="Output directory for embeddings"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model_name=args.model,
        device=args.device
    )
    
    # Load model
    generator.load_model()
    
    # Process citation data
    result = generator.process_citation_data(
        nodes_csv=args.nodes_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    print("\nEmbedding generation complete!")
    print(f"Shape: {result['shape']}")
    print(f"Saved to: {result['embeddings_file']}")


if __name__ == "__main__":
    main()