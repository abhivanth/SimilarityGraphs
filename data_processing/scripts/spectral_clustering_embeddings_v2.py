import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import gc
import psutil
import os
from scipy import sparse


def log_memory_usage(step_name: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {step_name}: {memory_mb:.1f} MB")


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

        log_memory_usage("After initialization")

    def compute_cosine_similarity(self) -> np.ndarray:
        """Compute cosine similarity matrix between all embeddings."""
        logging.info("Computing cosine similarity matrix...")
        log_memory_usage("Before similarity computation")

        self.similarity_matrix = cosine_similarity(self.embeddings)

        log_memory_usage("After similarity computation")
        logging.info("Cosine similarity matrix computed")
        return self.similarity_matrix

    def get_knn_similarity_graph(self, k: int = 10) -> sparse.csr_matrix:
        """Get k-NN similarity graph as sparse matrix to save memory."""
        from sklearn.neighbors import NearestNeighbors

        logging.info(f"Creating k-NN similarity graph with k={k}")
        log_memory_usage("Before k-NN computation")

        # Use NearestNeighbors to find k nearest neighbors efficiently
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(self.embeddings)

        # Get distances and indices
        distances, indices = nbrs.kneighbors(self.embeddings)

        # Convert distances to similarities (cosine distance = 1 - cosine similarity)
        similarities = 1 - distances

        # Create sparse affinity matrix directly
        row_indices = []
        col_indices = []
        data = []

        for i in range(self.n_papers):
            for j_idx in range(1, k + 1):  # Skip first (self)
                neighbor_idx = indices[i, j_idx]
                similarity = similarities[i, j_idx]
                if similarity > 0:  # Only store positive similarities
                    # Make symmetric
                    row_indices.extend([i, neighbor_idx])
                    col_indices.extend([neighbor_idx, i])
                    data.extend([similarity, similarity])

        # Create sparse matrix
        affinity_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_papers, self.n_papers)
        )

        # Ensure diagonal is 1 (self-similarity)
        affinity_matrix.setdiag(1.0)

        # Clean up intermediate variables
        del nbrs, distances, indices, similarities
        del row_indices, col_indices, data
        gc.collect()

        log_memory_usage("After k-NN graph creation")
        logging.info(f"Created sparse k-NN similarity graph with {affinity_matrix.nnz} non-zero entries")
        return affinity_matrix

    def get_epsilon_similarity_graph(self, epsilon: float = 0.7) -> sparse.csr_matrix:
        """Get epsilon-neighborhood similarity graph as sparse matrix."""
        logging.info(f"Creating epsilon-neighborhood graph with ε={epsilon}")
        log_memory_usage("Before epsilon graph computation")

        # Compute similarities in chunks to avoid full matrix
        chunk_size = min(1000, self.n_papers)
        row_indices = []
        col_indices = []
        data = []

        for i in range(0, self.n_papers, chunk_size):
            end_i = min(i + chunk_size, self.n_papers)
            chunk_embeddings = self.embeddings[i:end_i]

            # Compute similarities for this chunk
            chunk_similarities = cosine_similarity(chunk_embeddings, self.embeddings)

            # Find entries above threshold
            rows, cols = np.where(chunk_similarities >= epsilon)

            # Adjust row indices for chunk offset
            rows += i

            # Store non-zero entries
            for r, c in zip(rows, cols):
                if chunk_similarities[r - i, c] >= epsilon:
                    row_indices.append(r)
                    col_indices.append(c)
                    data.append(chunk_similarities[r - i, c])

            # Clean up chunk
            del chunk_embeddings, chunk_similarities
            gc.collect()

        # Create sparse matrix
        affinity_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_papers, self.n_papers)
        )

        # Clean up intermediate variables
        del row_indices, col_indices, data
        gc.collect()

        log_memory_usage("After epsilon graph creation")
        logging.info(f"Created sparse epsilon-neighborhood graph with {affinity_matrix.nnz} non-zero entries")
        return affinity_matrix

    def get_full_similarity_graph(self) -> sparse.csr_matrix:
        """Get full similarity graph as sparse matrix (threshold very low values)."""
        if self.similarity_matrix is None:
            self.compute_cosine_similarity()

        logging.info("Creating full similarity graph")
        log_memory_usage("Before full graph creation")

        # Convert to sparse by thresholding very small values
        threshold = 0.01  # Remove very small similarities to save memory
        similarity_sparse = sparse.csr_matrix(
            np.where(self.similarity_matrix >= threshold, self.similarity_matrix, 0)
        )

        # Convert similarity to affinity (ensure positive values)
        similarity_sparse.data = (similarity_sparse.data + 1) / 2  # Scale to [0, 1]

        # Clean up dense matrix
        del self.similarity_matrix
        self.similarity_matrix = None
        gc.collect()

        log_memory_usage("After full graph creation")
        logging.info(f"Created sparse full similarity graph with {similarity_sparse.nnz} non-zero entries")
        return similarity_sparse

    def get_graph_statistics(self, graph_type: str = "knn") -> Dict[str, Any]:
        """Get statistics about the similarity graph."""
        stats = {
            'n_papers': self.n_papers,
            'embedding_dim': self.embeddings.shape[1],
            'graph_type': graph_type
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

    def fit_predict(self, affinity_matrix: sparse.csr_matrix) -> np.ndarray:
        """
        Fit spectral clustering and predict clusters.

        Args:
            affinity_matrix: Precomputed sparse affinity matrix

        Returns:
            Cluster labels
        """
        logging.info("Fitting spectral clustering...")
        log_memory_usage("Before spectral clustering")

        # Aggressive memory cleanup before clustering
        gc.collect()

        self.clustering_model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            eigen_solver='lobpcg',  # More memory efficient
            assign_labels='kmeans',  # Keep for quality
            random_state=self.random_state,
            n_jobs=1,  # Avoid parallel memory overhead,
            verbose=True
        )

        self.cluster_labels = self.clustering_model.fit_predict(affinity_matrix)

        log_memory_usage("After spectral clustering")
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
        log_memory_usage("Before evaluation")

        # Silhouette Score
        logging.info("Computing silhouette score...")
        self.silhouette_avg = silhouette_score(embeddings, self.cluster_labels)
        metrics['silhouette_score'] = self.silhouette_avg
        logging.info(f"Silhouette score: {self.silhouette_avg:.4f}")

        # Adjusted Rand Index (if ground truth available)
        if ground_truth_labels is not None:
            logging.info("Computing Adjusted Rand Index...")
            self.plot_distribution_comparison(ground_truth_labels, self.cluster_labels)
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

        log_memory_usage("After evaluation")
        return metrics

    def plot_distribution_comparison(self, ground_truth_labels, predicted_labels,
                                     title="Ground Truth vs Predicted Distribution"):
        """
        Plot comparison between ground truth class distribution and predicted cluster distribution.

        Args:
            ground_truth_labels: Array of ground truth class labels
            predicted_labels: Array of predicted cluster labels
            title: Plot title
        """
        log_memory_usage("Before distribution plot")

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

        # Save plot
        save_path = "results/text_similarity_clustering/comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution comparison plot saved to {save_path}")

        plt.close()

        # Clean up
        del fig, ax1, ax2, bars1, bars2
        gc.collect()
        log_memory_usage("After distribution plot")

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

    def analyze_cluster_composition(self, ground_truth_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the composition of each cluster - which class labels are present.

        Args:
            ground_truth_labels: Array of ground truth class labels

        Returns:
            Dictionary with detailed cluster composition analysis
        """
        if self.cluster_labels is None or ground_truth_labels is None:
            raise ValueError("Must have both cluster labels and ground truth")

        logging.info("Analyzing cluster composition...")

        cluster_composition = {}
        unique_clusters = np.unique(self.cluster_labels)
        unique_classes = np.unique(ground_truth_labels)

        # Create detailed composition for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_ground_truth = ground_truth_labels[cluster_mask]

            # Count each class in this cluster
            class_counts = {}
            for class_label in unique_classes:
                count = np.sum(cluster_ground_truth == class_label)
                if count > 0:
                    class_counts[int(class_label)] = int(count)

            # Calculate percentages
            total_in_cluster = len(cluster_ground_truth)
            class_percentages = {class_id: (count / total_in_cluster) * 100
                                 for class_id, count in class_counts.items()}

            cluster_composition[int(cluster_id)] = {
                'total_papers': total_in_cluster,
                'class_counts': class_counts,
                'class_percentages': class_percentages,
                'num_different_classes': len(class_counts),
                'dominant_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None,
                'dominant_class_percentage': max(class_percentages.values()) if class_percentages else 0
            }

        # Create summary statistics
        summary_stats = {
            'total_clusters': len(unique_clusters),
            'total_classes': len(unique_classes),
            'avg_classes_per_cluster': np.mean([comp['num_different_classes']
                                                for comp in cluster_composition.values()]),
            'avg_dominant_class_percentage': np.mean([comp['dominant_class_percentage']
                                                      for comp in cluster_composition.values()]),
            'clusters_with_single_class': sum(1 for comp in cluster_composition.values()
                                              if comp['num_different_classes'] == 1),
            'clusters_with_multiple_classes': sum(1 for comp in cluster_composition.values()
                                                  if comp['num_different_classes'] > 1)
        }

        # Print detailed analysis
        self._print_cluster_composition_analysis(cluster_composition, summary_stats)

        # Save to CSV for detailed analysis
        self._save_cluster_composition_to_csv(cluster_composition)

        composition_analysis = {
            'cluster_composition': cluster_composition,
            'summary_statistics': summary_stats
        }

        logging.info("Cluster composition analysis completed")
        return composition_analysis

    def _print_cluster_composition_analysis(self, cluster_composition: Dict, summary_stats: Dict):
        """Print detailed cluster composition analysis."""
        print("\n" + "=" * 80)
        print("CLUSTER COMPOSITION ANALYSIS")
        print("=" * 80)

        print(f"Total clusters: {summary_stats['total_clusters']}")
        print(f"Total classes: {summary_stats['total_classes']}")
        print(f"Average classes per cluster: {summary_stats['avg_classes_per_cluster']:.2f}")
        print(f"Average dominant class percentage: {summary_stats['avg_dominant_class_percentage']:.1f}%")
        print(f"Clusters with single class: {summary_stats['clusters_with_single_class']}")
        print(f"Clusters with multiple classes: {summary_stats['clusters_with_multiple_classes']}")

        print("\n" + "-" * 80)
        print("DETAILED CLUSTER BREAKDOWN")
        print("-" * 80)

        for cluster_id in sorted(cluster_composition.keys()):
            comp = cluster_composition[cluster_id]
            print(f"\nCluster {cluster_id} ({comp['total_papers']} papers):")
            print(f"  Classes present: {comp['num_different_classes']}")
            print(f"  Dominant class: {comp['dominant_class']} ({comp['dominant_class_percentage']:.1f}%)")

            # Show top 3 classes in this cluster
            sorted_classes = sorted(comp['class_counts'].items(), key=lambda x: x[1], reverse=True)
            print("  Class distribution:")
            for class_id, count in sorted_classes[:3]:  # Show top 3
                percentage = comp['class_percentages'][class_id]
                print(f"    Class {class_id}: {count} papers ({percentage:.1f}%)")

            if len(sorted_classes) > 3:
                print(f"    ... and {len(sorted_classes) - 3} more classes")

    def _save_cluster_composition_to_csv(self, cluster_composition: Dict):
        """Save detailed cluster composition to CSV file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs("results/text_similarity_clustering", exist_ok=True)

            # Prepare data for CSV
            csv_data = []
            for cluster_id, comp in cluster_composition.items():
                for class_id, count in comp['class_counts'].items():
                    csv_data.append({
                        'cluster_id': cluster_id,
                        'class_id': class_id,
                        'paper_count': count,
                        'percentage_in_cluster': comp['class_percentages'][class_id],
                        'total_cluster_size': comp['total_papers'],
                        'is_dominant_class': (class_id == comp['dominant_class'])
                    })

            # Save to CSV
            df = pd.DataFrame(csv_data)
            csv_path = "results/text_similarity_clustering/cluster_composition_detailed.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Detailed cluster composition saved to {csv_path}")
            print(f"Detailed cluster composition saved to {csv_path}")

        except Exception as e:
            logging.warning(f"Could not save cluster composition to CSV: {e}")

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

    def plot_confusion_matrix(self,
                              ground_truth: np.ndarray,
                              predicted: np.ndarray,
                              title: str = "Clustering vs Ground Truth",
                              save_name: str = "confusion_matrix.png") -> None:
        """Plot confusion matrix between predicted clusters and ground truth."""
        log_memory_usage("Before confusion matrix")

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

        # Clean up
        del df_confusion, confusion_subset
        gc.collect()

        logging.info(f"Confusion matrix saved to {save_path}")
        log_memory_usage("After confusion matrix")

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
        log_memory_usage("Before loading embeddings")

        embeddings_path = Path(self.embeddings_file)
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        # Load the .npy file: [embedding_dims, node_id, class_idx]
        data = np.load(self.embeddings_file)
        log_memory_usage("After loading .npy file")

        # Extract components
        self.embeddings = data[:, :-2]  # All columns except last two
        self.node_ids = data[:, -2].astype(int).tolist()  # Second to last column
        self.class_labels = data[:, -1].astype(int)  # Last column

        # Clean up original data
        del data
        gc.collect()

        logging.info(f"Loaded {len(self.embeddings)} embeddings")
        logging.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        logging.info(f"Number of unique classes: {len(np.unique(self.class_labels))}")
        logging.info(f"Class range: {self.class_labels.min()} - {self.class_labels.max()}")

        # Validate data
        assert len(self.embeddings) == len(self.node_ids) == len(self.class_labels), \
            "Mismatch in data lengths"

        log_memory_usage("After loading embeddings")

    def run_clustering(self) -> Dict[str, Any]:
        """Run complete clustering pipeline with evaluation."""
        logging.info("Starting embedding-based spectral clustering pipeline...")
        log_memory_usage("Pipeline start")

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

        # Clean up similarity graph object to save memory
        del self.similarity_graph.similarity_matrix
        gc.collect()

        # Run spectral clustering
        logging.info("Running spectral clustering...")
        self.clustering_pipeline = SpectralClusteringPipeline(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        cluster_labels = self.clustering_pipeline.fit_predict(affinity_matrix)

        # Clean up affinity matrix
        del affinity_matrix
        gc.collect()
        log_memory_usage("After clustering")

        # Evaluate clustering
        metrics = self.clustering_pipeline.evaluate_clustering(
            self.embeddings, self.class_labels
        )

        # Analyze cluster purity
        purity_analysis = self.clustering_pipeline.analyze_cluster_purity(self.class_labels)

        # NEW: Analyze cluster composition (which class labels are in each cluster)
        composition_analysis = self.clustering_pipeline.analyze_cluster_composition(self.class_labels)

        # Get results
        graph_stats = self.similarity_graph.get_graph_statistics(self.graph_type)
        cluster_info = self.clustering_pipeline.get_cluster_info()

        # Create visualizations (skip t-SNE)
        logging.info("Creating visualizations...")

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
            'composition_analysis': composition_analysis,  # NEW: Added composition analysis
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        self.print_results_formatted(results)

        log_memory_usage("Pipeline end")
        logging.info("Clustering pipeline completed successfully!")
        return results

    def print_results_formatted(self, results_clustering: dict):
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

        print(f"\nCOMPOSITION SUMMARY:")
        composition = results_clustering['composition_analysis']['summary_statistics']
        print(f"  Clusters with single class: {composition['clusters_with_single_class']}")
        print(f"  Clusters with multiple classes: {composition['clusters_with_multiple_classes']}")
        print(f"  Average classes per cluster: {composition['avg_classes_per_cluster']:.2f}")
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
        default="results/text_similarity_clustering",
        help="Output directory for results"
    )

    args = parser.parse_args()

    runner = EmbeddingSpectralClusteringRunner(
        embeddings_file=args.embeddings_file,
        n_clusters=args.n_clusters,
        graph_type=args.graph_type,
        k_neighbors=args.k_neighbors,
        epsilon=args.epsilon,
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