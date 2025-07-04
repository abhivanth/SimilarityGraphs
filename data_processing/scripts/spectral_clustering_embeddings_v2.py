import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


class EmbeddingSimilarityGraph:
    """Create similarity graphs from embeddings."""

    def __init__(self, embeddings: np.ndarray, node_ids: Optional[list] = None, class_labels: Optional[list] = None):
        """
        Initialize with embeddings.

        Args:
            embeddings: Array of shape (n_papers, embedding_dim)
            node_ids: Optional list of node IDs
            class_labels: Optional list of ground truth class labels
        """
        self.embeddings = embeddings
        self.node_ids = node_ids or list(range(len(embeddings)))
        self.class_labels = class_labels
        self.n_papers = len(embeddings)
        self.similarity_matrix = None

        logging.info(f"Initialized similarity graph with {self.n_papers} papers")
        logging.info(f"Embedding dimension: {embeddings.shape[1]}")
        if class_labels is not None:
            logging.info(f"Ground truth classes: {len(set(class_labels))} unique classes")

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

    def get_epsilon_similarity_graph(self, epsilon: float = 0.7) -> np.ndarray:
        """Get epsilon-neighborhood similarity graph."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()

        logging.info(f"Creating epsilon-neighborhood graph with ε={epsilon}")

        # Create binary adjacency matrix based on threshold
        affinity_matrix = np.where(self.similarity_matrix >= epsilon,
                                   self.similarity_matrix, 0)

        # Ensure diagonal is 1 (self-similarity)
        np.fill_diagonal(affinity_matrix, 1.0)

        # Count edges
        n_edges = np.count_nonzero(affinity_matrix) - self.n_papers  # Subtract diagonal
        logging.info(f"Created epsilon-neighborhood graph with {n_edges} edges")

        return affinity_matrix

    def get_full_similarity_graph(self) -> np.ndarray:
        """Get full similarity graph (all pairwise similarities)."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()

        # Convert similarity to affinity (ensure positive values)
        affinity_matrix = (self.similarity_matrix + 1) / 2  # Scale to [0, 1]
        logging.info("Created full similarity graph")
        return affinity_matrix

    def get_graph_statistics(self, graph_type: str = "knn") -> Dict[str, Any]:
        """Get statistics about the similarity graph."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()

        stats = {
            'n_papers': self.n_papers,
            'embedding_dim': self.embeddings.shape[1],
            'graph_type': graph_type,
            'mean_similarity': float(np.mean(self.similarity_matrix)),
            'std_similarity': float(np.std(self.similarity_matrix)),
            'min_similarity': float(np.min(self.similarity_matrix)),
            'max_similarity': float(np.max(self.similarity_matrix))
        }

        if self.class_labels is not None:
            stats['n_ground_truth_classes'] = len(set(self.class_labels))
            stats['class_distribution'] = dict(zip(*np.unique(self.class_labels, return_counts=True)))

        return stats


class SpectralClusteringPipeline:
    """Complete spectral clustering pipeline with evaluation against ground truth."""

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
        self.ari_score = None

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

    def evaluate_clustering(self, embeddings: np.ndarray, ground_truth_labels: Optional[np.ndarray] = None) -> Dict[
        str, float]:
        """
        Evaluate clustering using silhouette score and ARI (if ground truth available).

        Args:
            embeddings: Original embeddings for evaluation
            ground_truth_labels: Ground truth class labels for ARI calculation

        Returns:
            Dictionary with evaluation metrics
        """
        if self.cluster_labels is None:
            raise ValueError("Must fit clustering first")

        metrics = {}

        # Silhouette Score
        logging.info("Computing silhouette score...")
        self.silhouette_avg = silhouette_score(embeddings, self.cluster_labels)
        metrics['silhouette_score'] = self.silhouette_avg
        logging.info(f"Silhouette score: {self.silhouette_avg:.4f}")

        # Adjusted Rand Index (if ground truth available)
        if ground_truth_labels is not None:
            logging.info("Computing Adjusted Rand Index...")
            self.plot_distribution_comparison(ground_truth_labels,self.cluster_labels)
            self.ari_score = adjusted_rand_score(ground_truth_labels, self.cluster_labels)
            metrics['adjusted_rand_index'] = self.ari_score
            logging.info(f"Adjusted Rand Index: {self.ari_score:.4f}")

            # Additional evaluation metrics
            from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score

            metrics['normalized_mutual_info'] = normalized_mutual_info_score(ground_truth_labels, self.cluster_labels)
            metrics['homogeneity_score'] = homogeneity_score(ground_truth_labels, self.cluster_labels)
            metrics['completeness_score'] = completeness_score(ground_truth_labels, self.cluster_labels)

            logging.info(f"NMI: {metrics['normalized_mutual_info']:.4f}")
            logging.info(f"Homogeneity: {metrics['homogeneity_score']:.4f}")
            logging.info(f"Completeness: {metrics['completeness_score']:.4f}")

        return metrics

    def plot_distribution_comparison(self, ground_truth_labels, predicted_labels,
                                     title="Ground Truth vs Predicted Distribution"):
        """
        Plot comparison between ground truth class distribution and predicted cluster distribution.

        Args:
            ground_truth_labels: Array of ground truth class labels
            predicted_labels: Array of predicted cluster labels
            title: Plot title
            save_path: Optional path to save the plot
        """

        # Get distributions
        gt_unique, gt_counts = np.unique(ground_truth_labels, return_counts=True)
        pred_unique, pred_counts = np.unique(predicted_labels, return_counts=True)

        # Create side-by-side bar charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Ground truth distribution
        bars1 = ax1.bar(gt_unique, gt_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Ground Truth Class Distribution')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars1, gt_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(gt_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        # Predicted cluster distribution
        bars2 = ax2.bar(pred_unique, pred_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_title('Predicted Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Papers')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, count in zip(bars2, pred_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pred_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        # Add statistics text
        ax1.text(0.02, 0.98, f'Classes: {len(gt_unique)}\nTotal: {sum(gt_counts)}',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax2.text(0.02, 0.98, f'Clusters: {len(pred_unique)}\nTotal: {sum(pred_counts)}',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save or show plot
        save_path = "results/text_similarity_clustering/comparison.png"
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution comparison plot saved to {save_path}")

        plt.close()

    def analyze_cluster_purity(self, ground_truth_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze cluster purity against ground truth."""
        if self.cluster_labels is None or ground_truth_labels is None:
            raise ValueError("Must have both cluster labels and ground truth")

        cluster_purity = []
        unique_clusters = np.unique(self.cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_ground_truth = ground_truth_labels[cluster_mask]

            # Find most common ground truth class in this cluster
            most_common_class = np.bincount(cluster_ground_truth.astype(int)).argmax()
            purity = np.mean(cluster_ground_truth == most_common_class)

            cluster_purity.append({
                'cluster_id': int(cluster_id),
                'size': int(np.sum(cluster_mask)),
                'dominant_class': int(most_common_class),
                'purity': float(purity)
            })

        # Overall purity statistics
        purities = [cp['purity'] for cp in cluster_purity]
        purity_stats = {
            'cluster_purities': cluster_purity,
            'mean_purity': float(np.mean(purities)),
            'std_purity': float(np.std(purities)),
            'min_purity': float(np.min(purities)),
            'max_purity': float(np.max(purities))
        }

        return purity_stats

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about the clustering results."""
        if self.cluster_labels is None:
            raise ValueError("Must fit clustering first")

        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)

        cluster_info = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': dict(zip(unique_labels.tolist(), counts.tolist())),
            'silhouette_score': self.silhouette_avg,
            'adjusted_rand_index': self.ari_score,
            'largest_cluster_size': int(np.max(counts)),
            'smallest_cluster_size': int(np.min(counts)),
            'mean_cluster_size': float(np.mean(counts)),
            'std_cluster_size': float(np.std(counts))
        }

        return cluster_info


class ClusteringVisualizer:
    """Visualize clustering results with ground truth comparison."""

    def __init__(self, output_dir: str = "results"):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_tsne_clusters(self,
                           embeddings: np.ndarray,
                           cluster_labels: np.ndarray,
                           ground_truth_labels: Optional[np.ndarray] = None,
                           title: str = "t-SNE Visualization of Clusters",
                           save_name: str = "tsne_clusters.png") -> None:
        """
        Create t-SNE visualization of clusters with optional ground truth comparison.
        """
        logging.info("Creating t-SNE visualization...")

        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create subplots
        if ground_truth_labels is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Predicted clusters
            scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=cluster_labels, cmap='tab10', alpha=0.7)
            ax1.set_title(f"{title} - Predicted Clusters")
            ax1.set_xlabel('t-SNE Component 1')
            ax1.set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter1, ax=ax1)

            # Ground truth
            scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=ground_truth_labels, cmap='tab20', alpha=0.7)
            ax2.set_title(f"{title} - Ground Truth Classes")
            ax2.set_xlabel('t-SNE Component 1')
            ax2.set_ylabel('t-SNE Component 2')
            plt.colorbar(scatter2, ax=ax2)
        else:
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

    def plot_confusion_matrix(self,
                              ground_truth: np.ndarray,
                              predicted: np.ndarray,
                              title: str = "Clustering vs Ground Truth",
                              save_name: str = "confusion_matrix.png") -> None:
        """Plot confusion matrix between predicted clusters and ground truth."""

        # Create confusion matrix
        df_confusion = pd.crosstab(ground_truth, predicted, margins=True)

        plt.figure(figsize=(12, 10))

        # Plot heatmap (excluding margins for clarity)
        confusion_subset = df_confusion.iloc[:-1, :-1]  # Remove 'All' row/column
        sns.heatmap(confusion_subset, annot=False, cmap='Blues', fmt='d')

        plt.title(title)
        plt.xlabel('Predicted Cluster')
        plt.ylabel('Ground Truth Class')

        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Confusion matrix saved to {save_path}")

    def plot_cluster_sizes(self,
                           cluster_info: Dict[str, Any],
                           title: str = "Cluster Size Distribution",
                           save_name: str = "cluster_sizes.png") -> None:
        """Plot cluster size distribution."""
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

    def plot_evaluation_metrics(self,
                                metrics: Dict[str, float],
                                title: str = "Clustering Evaluation Metrics",
                                save_name: str = "evaluation_metrics.png") -> None:
        """Plot evaluation metrics as a bar chart."""

        # Filter out non-numeric metrics for plotting
        plot_metrics = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float)) and k != 'n_ground_truth_classes'}

        plt.figure(figsize=(10, 6))
        metric_names = list(plot_metrics.keys())
        metric_values = list(plot_metrics.values())

        bars = plt.bar(metric_names, metric_values, alpha=0.7,
                       color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'plum'])
        plt.xlabel('Evaluation Metric')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom')

        # Save plot
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Evaluation metrics plot saved to {save_path}")


class EmbeddingSpectralClusteringRunner:
    """Main runner for embedding-based spectral clustering with ground truth evaluation."""

    def __init__(self,
                 embeddings_file: str,
                 n_clusters: int = 40,
                 graph_type: str = "knn",
                 k_neighbors: int = 20,
                 epsilon: float = 0.7,
                 output_dir: str = "results/text_similarity_clustering",
                 random_state: int = 42):
        """
        Initialize clustering runner.

        Args:
            embeddings_file: Path to embeddings .npy file with labels
            n_clusters: Number of clusters (should match number of arXiv categories)
            graph_type: Type of similarity graph ('knn', 'epsilon', 'full')
            k_neighbors: Number of neighbors for k-NN graph
            epsilon: Threshold for epsilon-neighborhood graph
            output_dir: Output directory for results
            random_state: Random state for reproducibility
        """
        self.embeddings_file = embeddings_file
        self.n_clusters = n_clusters
        self.graph_type = graph_type
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Initialize components
        self.embeddings = None
        self.node_ids = None
        self.class_labels = None
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

    def load_embeddings_with_labels(self):
        """Load embeddings with class labels from .npy file."""
        logging.info(f"Loading embeddings with labels from {self.embeddings_file}")

        embeddings_path = Path(self.embeddings_file)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        # Load the .npy file: [embedding_dims, node_id, class_idx]
        data = np.load(self.embeddings_file)

        # Extract components
        self.embeddings = data[:, :-2]  # All columns except last two
        self.node_ids = data[:, -2].astype(int).tolist()  # Second to last column
        self.class_labels = data[:, -1].astype(int)  # Last column

        logging.info(f"Loaded {len(self.embeddings)} embeddings")
        logging.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        logging.info(f"Number of unique classes: {len(np.unique(self.class_labels))}")
        logging.info(f"Class range: {self.class_labels.min()} - {self.class_labels.max()}")

        # Validate data
        assert len(self.embeddings) == len(self.node_ids) == len(self.class_labels), \
            "Mismatch in data lengths"

    def run_clustering(self) -> Dict[str, Any]:
        """Run complete clustering pipeline with evaluation."""
        logging.info("Starting embedding-based spectral clustering pipeline...")

        # Load embeddings with labels
        self.load_embeddings_with_labels()

        # Create similarity graph
        logging.info(f"Creating {self.graph_type} similarity graph...")
        self.similarity_graph = EmbeddingSimilarityGraph(
            self.embeddings, self.node_ids, self.class_labels
        )

        # Get affinity matrix based on graph type
        if self.graph_type == "knn":
            affinity_matrix = self.similarity_graph.get_knn_similarity_graph(k=self.k_neighbors)
        elif self.graph_type == "epsilon":
            affinity_matrix = self.similarity_graph.get_epsilon_similarity_graph(epsilon=self.epsilon)
        elif self.graph_type == "full":
            affinity_matrix = self.similarity_graph.get_full_similarity_graph()
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        # Run spectral clustering
        logging.info("Running spectral clustering...")
        self.clustering_pipeline = SpectralClusteringPipeline(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        cluster_labels = self.clustering_pipeline.fit_predict(affinity_matrix)

        # Evaluate clustering
        metrics = self.clustering_pipeline.evaluate_clustering(
            self.embeddings, self.class_labels
        )

        # Analyze cluster purity
        purity_analysis = self.clustering_pipeline.analyze_cluster_purity(self.class_labels)

        # Get results
        graph_stats = self.similarity_graph.get_graph_statistics(self.graph_type)
        cluster_info = self.clustering_pipeline.get_cluster_info()

        # Create visualizations
        logging.info("Creating visualizations...")

        # t-SNE visualization
        self.visualizer.plot_tsne_clusters(
            self.embeddings,
            cluster_labels,
            self.class_labels,
            title=f"t-SNE: {self.graph_type.upper()} Similarity Graph Clustering"
        )

        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            self.class_labels,
            cluster_labels,
            title="Predicted Clusters vs Ground Truth Classes"
        )

        # Cluster sizes
        self.visualizer.plot_cluster_sizes(
            cluster_info,
            title=f"Cluster Sizes - {self.graph_type.upper()} Graph"
        )

        # Evaluation metrics
        self.visualizer.plot_evaluation_metrics(
            metrics,
            title=f"Evaluation Metrics - {self.graph_type.upper()} Graph"
        )

        # Prepare comprehensive results
        results = {
            'method': 'embedding_spectral_clustering',
            'embeddings_file': str(self.embeddings_file),
            'graph_type': self.graph_type,
            'graph_parameters': {
                'k_neighbors': self.k_neighbors if self.graph_type == 'knn' else None,
                'epsilon': self.epsilon if self.graph_type == 'epsilon' else None
            },
            'n_clusters': self.n_clusters,
            'n_papers': len(self.embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'evaluation_metrics': metrics,
            'graph_statistics': graph_stats,
            'cluster_info': cluster_info,
            'purity_analysis': purity_analysis,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        self.print_results_formatted(results)

        logging.info("Clustering pipeline completed successfully!")
        return results

    def print_results_formatted(self,results_clustering : dict):
        print("=" * 60)
        print("SPECTRAL CLUSTERING RESULTS")
        print("=" * 60)

        print(f"Method: {results_clustering['method']}")
        print(f"Graph type: {results_clustering['graph_type']}")
        print(f"Number of papers: {results_clustering['n_papers']}")
        print(f"Number of clusters: {results_clustering['n_clusters']}")

        print(f"\nEVALUATION METRICS:")
        for metric, value in results_clustering['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nPURITY ANALYSIS:")
        purity = results_clustering['purity_analysis']
        print(f"  Mean purity: {purity['mean_purity']:.4f}")
        print(f"  Max purity: {purity['max_purity']:.4f}")
        print("=" * 60)


def main():
    """Main function to run embedding-based spectral clustering."""
    import argparse

    parser = argparse.ArgumentParser(description="Run spectral clustering on embeddings with ground truth evaluation")
    parser.add_argument(
        "--embeddings-file",
        type=str,
        required=True,
        help="Path to embeddings .npy file with labels [embedding_dims, node_id, class_idx]"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=40,
        help="Number of clusters (default: 40 for arXiv categories)"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["knn", "epsilon", "full"],
        default="knn",
        help="Type of similarity graph to create"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=20,
        help="Number of neighbors for k-NN graph"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7,
        help="Threshold for epsilon-neighborhood graph"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/text_similarity_clustering")

    args = parser.parse_args()

    runner = EmbeddingSpectralClusteringRunner(
        embeddings_file=args.embeddings_file,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir
    )

    results = runner.run_clustering()

    # Print summary
    print("\n" + "=" * 60)
    print("EMBEDDING-BASED SPECTRAL CLUSTERING COMPLETED")
    print("=" * 60)
    print(f"Silhouette Score: {results['evaluation_metrics']['silhouette_score']:.4f}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Number of papers: {results['n_papers']}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()