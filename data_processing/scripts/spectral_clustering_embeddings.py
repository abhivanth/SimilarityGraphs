import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class EmbeddingSimilarityGraph:
    """Create similarity graphs from embeddings."""
    
    def __init__(self, embeddings: np.ndarray, paper_ids: Optional[list] = None):
        """
        Initialize with embeddings.
        
        Args:
            embeddings: Array of shape (n_papers, embedding_dim)
            paper_ids: Optional list of paper IDs
        """
        self.embeddings = embeddings
        self.paper_ids = paper_ids or list(range(len(embeddings)))
        self.n_papers = len(embeddings)
        self.similarity_matrix = None
        
        logging.info(f"Initialized similarity graph with {self.n_papers} papers")
        logging.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    def compute_cosine_similarity(self) -> np.ndarray:
        """Compute cosine similarity matrix between all embeddings."""
        logging.info("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        logging.info("Cosine similarity matrix computed")
        return self.similarity_matrix

    def get_knn_similarity_graph(self, k: int = 10) -> np.ndarray:
        """Get k-NN similarity graph to save memory."""
        from sklearn.neighbors import NearestNeighbors

        logging.info(f"Creating k-NN similarity graph with k={k}")

        # Use NearestNeighbors to find k nearest neighbors efficiently
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(self.embeddings)

        # Get distances and indices
        distances, indices = nbrs.kneighbors(self.embeddings)

        # Convert distances to similarities (cosine distance = 1 - cosine similarity)
        similarities = 1 - distances

        # Create sparse affinity matrix
        from scipy.sparse import lil_matrix
        affinity_matrix = lil_matrix((self.n_papers, self.n_papers))

        for i in range(self.n_papers):
            for j_idx in range(1, k + 1):  # Skip first (self)
                neighbor_idx = indices[i, j_idx]
                similarity = similarities[i, j_idx]
                # Make symmetric
                affinity_matrix[i, neighbor_idx] = max(0, similarity)
                affinity_matrix[neighbor_idx, i] = max(0, similarity)

        # Convert to dense for spectral clustering
        dense_matrix = affinity_matrix.toarray()
        logging.info("Created k-NN similarity graph")
        return dense_matrix
    
    def get_full_similarity_graph(self) -> np.ndarray:
        """Get full similarity graph (all pairwise similarities)."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()
        
        # Convert similarity to affinity (ensure positive values)
        affinity_matrix = (self.similarity_matrix + 1) / 2  # Scale to [0, 1]
        logging.info("Created full similarity graph")
        return affinity_matrix
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the similarity graph."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()
            
        stats = {
            'n_papers': self.n_papers,
            'embedding_dim': self.embeddings.shape[1],
            'mean_similarity': float(np.mean(self.similarity_matrix)),
            'std_similarity': float(np.std(self.similarity_matrix)),
            'min_similarity': float(np.min(self.similarity_matrix)),
            'max_similarity': float(np.max(self.similarity_matrix))
        }
        
        return stats


class SpectralClusteringPipeline:
    """Complete spectral clustering pipeline."""
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Initialize clustering pipeline.
        
        Args:
            n_clusters: Number of clusters to find
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clustering_model = None
        self.cluster_labels = None
        self.silhouette_avg = None
        
        logging.info(f"Initialized spectral clustering with {n_clusters} clusters")
    
    def fit_predict(self, affinity_matrix: np.ndarray) -> np.ndarray:
        """
        Fit spectral clustering and predict clusters.
        
        Args:
            affinity_matrix: Precomputed affinity matrix
            
        Returns:
            Cluster labels
        """
        logging.info("Fitting spectral clustering...")
        
        self.clustering_model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.cluster_labels = self.clustering_model.fit_predict(affinity_matrix)
        logging.info("Spectral clustering completed")
        
        return self.cluster_labels
    
    def evaluate_clustering(self, embeddings: np.ndarray) -> float:
        """
        Evaluate clustering using silhouette score.
        
        Args:
            embeddings: Original embeddings for evaluation
            
        Returns:
            Silhouette score
        """
        if self.cluster_labels is None:
            raise ValueError("Must fit clustering first")
            
        logging.info("Computing silhouette score...")
        self.silhouette_avg = silhouette_score(embeddings, self.cluster_labels)
        logging.info(f"Silhouette score: {self.silhouette_avg:.4f}")
        
        return self.silhouette_avg
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering results."""
        if self.cluster_labels is None:
            raise ValueError("Must fit clustering first")
            
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        
        cluster_info = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': dict(zip(unique_labels.tolist(), counts.tolist())),
            'silhouette_score': self.silhouette_avg,
            'largest_cluster_size': int(np.max(counts)),
            'smallest_cluster_size': int(np.min(counts)),
            'mean_cluster_size': float(np.mean(counts)),
            'std_cluster_size': float(np.std(counts))
        }
        
        return cluster_info


class ClusteringVisualizer:
    """Visualize clustering results."""
    
    def __init__(self, output_dir: str = "results"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_tsne_clusters(self, 
                          embeddings: np.ndarray, 
                          cluster_labels: np.ndarray,
                          title: str = "t-SNE Visualization of Clusters",
                          save_name: str = "tsne_clusters.png") -> None:
        """
        Create t-SNE visualization of clusters.
        
        Args:
            embeddings: Original embeddings
            cluster_labels: Cluster assignments
            title: Plot title
            save_name: Filename to save plot
        """
        logging.info("Creating t-SNE visualization...")
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"t-SNE plot saved to {save_path}")
    
    def plot_cluster_sizes(self, 
                          cluster_info: Dict[str, Any],
                          title: str = "Cluster Size Distribution",
                          save_name: str = "cluster_sizes.png") -> None:
        """
        Plot cluster size distribution.
        
        Args:
            cluster_info: Cluster information dictionary
            title: Plot title
            save_name: Filename to save plot
        """
        cluster_sizes = cluster_info['cluster_sizes']
        
        plt.figure(figsize=(10, 6))
        clusters = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        plt.bar(clusters, sizes, alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Papers')
        plt.title(title)
        plt.xticks(clusters)
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            plt.text(clusters[i], v + 0.5, str(v), ha='center')
        
        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Cluster sizes plot saved to {save_path}")


class EmbeddingSpectralClusteringRunner:
    """Main runner for embedding-based spectral clustering."""
    
    def __init__(self, 
                 embeddings_file: str,
                 n_clusters: int = 8,
                 output_dir: str = "results/milestone1",
                 random_state: int = 42):
        """
        Initialize clustering runner.
        
        Args:
            embeddings_file: Path to embeddings .npy file
            n_clusters: Number of clusters
            output_dir: Output directory for results
            random_state: Random state for reproducibility
        """
        self.embeddings_file = embeddings_file
        self.n_clusters = n_clusters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize components
        self.embeddings = None
        self.paper_ids = None
        self.similarity_graph = None
        self.clustering_pipeline = None
        self.visualizer = ClusteringVisualizer(str(self.output_dir))
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"clustering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_embeddings(self):
        """Load embeddings from file."""
        logging.info(f"Loading embeddings from {self.embeddings_file}")
        
        embeddings_path = Path(self.embeddings_file)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        self.embeddings = np.load(self.embeddings_file)
        
        # Try to load corresponding CSV file with paper IDs
        csv_file = embeddings_path.with_suffix('.csv')
        if csv_file.exists():
            logging.info(f"Loading paper IDs from {csv_file}")
            df = pd.read_csv(csv_file)
            self.paper_ids = df['paper_id'].tolist()
        else:
            logging.warning("No CSV file found, using index as paper IDs")
            self.paper_ids = list(range(len(self.embeddings)))
        
        logging.info(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embeddings.shape[1]}")
    
    def run_clustering(self) -> Dict[str, Any]:
        """Run complete clustering pipeline."""
        logging.info("Starting embedding-based spectral clustering pipeline...")
        
        # Load embeddings
        self.load_embeddings()
        
        # Create similarity graph
        logging.info("Creating similarity graph...")
        self.similarity_graph = EmbeddingSimilarityGraph(self.embeddings, self.paper_ids)
        affinity_matrix = self.similarity_graph.get_knn_similarity_graph()
        
        # Run spectral clustering
        logging.info("Running spectral clustering...")
        self.clustering_pipeline = SpectralClusteringPipeline(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        
        cluster_labels = self.clustering_pipeline.fit_predict(affinity_matrix)
        silhouette_score = self.clustering_pipeline.evaluate_clustering(self.embeddings)
        
        # Get results
        graph_stats = self.similarity_graph.get_graph_statistics()
        cluster_info = self.clustering_pipeline.get_cluster_info()
        
        # Create visualizations
        logging.info("Creating visualizations...")
        self.visualizer.plot_tsne_clusters(
            self.embeddings, 
            cluster_labels,
            title="t-SNE Visualization of Embedding-based Clusters"
        )
        self.visualizer.plot_cluster_sizes(cluster_info)
        
        # Prepare results
        results = {
            'method': 'embedding_spectral_clustering',
            'graph_type': 'knn-similarity',
            'n_clusters': self.n_clusters,
            'n_papers': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'silhouette_score': silhouette_score,
            'graph_statistics': graph_stats,
            'cluster_info': cluster_info,
            'embeddings_file': str(self.embeddings_file),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(results, cluster_labels)
        
        logging.info("Clustering pipeline completed successfully!")
        return results

    def _save_results(self, results: Dict[str, Any], cluster_labels: np.ndarray):
        """Save clustering results to files."""

        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Convert results to JSON-serializable format
        json_results = convert_numpy_types(results)
        results_file = self.output_dir / "clustering_results.json"
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        logging.info(f"Results saved to {results_file}")
        
        # Save cluster assignments
        cluster_assignments = pd.DataFrame({
            'paper_id': self.paper_ids,
            'cluster': cluster_labels
        })
        
        assignments_file = self.output_dir / "cluster_assignments.csv"
        cluster_assignments.to_csv(assignments_file, index=False)
        logging.info(f"Cluster assignments saved to {assignments_file}")
        
        # Save summary
        summary_file = self.output_dir / "clustering_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Embedding-based Spectral Clustering Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Embeddings file: {self.embeddings_file}\n")
            f.write(f"Number of papers: {results['n_papers']}\n")
            f.write(f"Embedding dimension: {results['embedding_dim']}\n")
            f.write(f"Number of clusters: {results['n_clusters']}\n")
            f.write(f"Silhouette score: {results['silhouette_score']:.4f}\n\n")
            
            f.write("Graph Statistics:\n")
            for key, value in results['graph_statistics'].items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\nCluster Information:\n")
            for key, value in results['cluster_info'].items():
                f.write(f"  {key}: {value}\n")
        
        logging.info(f"Summary saved to {summary_file}")


def main():
    """Main function to run embedding-based spectral clustering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run spectral clustering on embeddings")
    parser.add_argument(
        "--embeddings-file",
        type=str,
        required=True,
        help="Path to embeddings .npy file"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of clusters"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/milestone1",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Run clustering
    runner = EmbeddingSpectralClusteringRunner(
        embeddings_file=args.embeddings_file,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
    
    results = runner.run_clustering()
    
    # Print summary
    print("\n" + "="*60)
    print("EMBEDDING-BASED SPECTRAL CLUSTERING COMPLETED")
    print("="*60)
    print(f"Silhouette Score: {results['silhouette_score']:.4f}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Number of papers: {results['n_papers']}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()