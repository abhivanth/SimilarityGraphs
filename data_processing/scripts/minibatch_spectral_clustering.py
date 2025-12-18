import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple, List
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


class StratifiedSampler:
    """Create stratified sample preserving class distribution."""

    def __init__(self, stratified_ratio: float = 0.1, random_state: int = 42):
        """
        Initialize stratified sampler.

        Args:
            stratified_ratio: Ratio of data to sample (e.g., 0.1 for 10%)
            random_state: Random state for reproducibility
        """
        self.stratified_ratio = stratified_ratio
        self.random_state = random_state

    def create_stratified_sample(self,
                                 embeddings: np.ndarray,
                                 node_ids: List[int],
                                 class_labels: np.ndarray) -> Tuple[np.ndarray, List[int], np.ndarray,
    np.ndarray, List[int], np.ndarray]:
        """
        Create stratified sample and return both stratified and remaining sets.

        Args:
            embeddings: Full embedding matrix
            node_ids: List of node IDs
            class_labels: Ground truth class labels

        Returns:
            Tuple of (stratified_embeddings, stratified_ids, stratified_labels,
                     remaining_embeddings, remaining_ids, remaining_labels)
        """
        logging.info(f"Creating stratified sample with ratio={self.stratified_ratio}")
        log_memory_usage("Before stratified sampling")

        # Create indices array
        indices = np.arange(len(embeddings))

        # Perform stratified split
        stratified_indices, remaining_indices = train_test_split(
            indices,
            test_size=(1 - self.stratified_ratio),
            stratify=class_labels,
            random_state=self.random_state
        )

        # Extract stratified subset
        stratified_embeddings = embeddings[stratified_indices]
        stratified_ids = [node_ids[i] for i in stratified_indices]
        stratified_labels = class_labels[stratified_indices]

        # Extract remaining subset
        remaining_embeddings = embeddings[remaining_indices]
        remaining_ids = [node_ids[i] for i in remaining_indices]
        remaining_labels = class_labels[remaining_indices]

        logging.info(f"Stratified sample size: {len(stratified_embeddings)}")
        logging.info(f"Remaining sample size: {len(remaining_embeddings)}")

        # Verify class distribution preservation
        original_dist = dict(zip(*np.unique(class_labels, return_counts=True)))
        stratified_dist = dict(zip(*np.unique(stratified_labels, return_counts=True)))

        logging.info("Original class distribution (counts): %s", original_dist)
        logging.info("Stratified class distribution (counts): %s", stratified_dist)

        log_memory_usage("After stratified sampling")

        return (stratified_embeddings, stratified_ids, stratified_labels,
                remaining_embeddings, remaining_ids, remaining_labels)


class MiniBatchSimilarityGraph:
    """Create similarity graphs from embeddings for mini-batch."""

    def __init__(self, embeddings: np.ndarray, node_ids: Optional[list] = None):
        """
        Initialize with embeddings.

        Args:
            embeddings: Array of shape (n_papers, embedding_dim)
            node_ids: Optional list of node IDs
        """
        self.embeddings = embeddings
        self.node_ids = node_ids or list(range(len(embeddings)))
        self.n_papers = len(embeddings)

        logging.info(f"Initialized mini-batch similarity graph with {self.n_papers} papers")
        logging.info(f"Embedding dimension: {embeddings.shape[1]}")
        log_memory_usage("After mini-batch graph initialization")

    def get_knn_similarity_graph(self, k: int = 3) -> sparse.csr_matrix:
        """Get k-NN similarity graph as sparse matrix."""
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


class MiniBatchSpectralClustering:
    """Spectral clustering for mini-batch with out-of-sample extension."""

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Initialize mini-batch spectral clustering.

        Args:
            n_clusters: Number of clusters to find
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clustering_model = None
        self.cluster_labels_stratified = None
        self.spectral_embeddings = None  # Store spectral embeddings from stratified subset
        self.cluster_centroids = None  # Centroids in spectral space

        logging.info(f"Initialized mini-batch spectral clustering with {n_clusters} clusters")

    def fit_stratified(self, affinity_matrix: sparse.csr_matrix) -> np.ndarray:
        """
        Fit spectral clustering on stratified subset and get spectral embeddings.

        Args:
            affinity_matrix: Precomputed sparse affinity matrix for stratified subset

        Returns:
            Cluster labels for stratified subset
        """
        logging.info("Fitting spectral clustering on stratified subset...")
        log_memory_usage("Before spectral clustering")

        # Aggressive memory cleanup before clustering
        gc.collect()

        self.clustering_model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            eigen_solver='lobpcg',  # More memory efficient
            assign_labels='kmeans',
            random_state=self.random_state,
            n_jobs=1,  # Avoid parallel memory overhead
            verbose=True
        )

        self.cluster_labels_stratified = self.clustering_model.fit_predict(affinity_matrix)

        # Extract spectral embeddings (eigenvectors) from the clustering model
        # The spectral embeddings are the result of the eigendecomposition
        # We need to access the internal embeddings used by kmeans
        self.spectral_embeddings = self.clustering_model.affinity_matrix_

        # Actually, we need to recompute to get the actual spectral coordinates
        # Let's extract them properly
        from sklearn.cluster import KMeans
        from scipy.sparse.linalg import eigsh
        from sklearn.preprocessing import normalize

        # Recompute spectral embeddings for clarity
        logging.info("Computing spectral embeddings explicitly...")

        # Compute normalized Laplacian
        D = np.array(affinity_matrix.sum(axis=1)).flatten()
        D[D == 0] = 1  # Avoid division by zero
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(D))
        L_norm = sparse.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt

        # Compute eigenvectors
        eigenvalues, eigenvectors = eigsh(L_norm, k=self.n_clusters, which='SM')

        # Normalize eigenvectors
        self.spectral_embeddings = normalize(eigenvectors)

        # Recompute cluster labels using kmeans on spectral embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.cluster_labels_stratified = kmeans.fit_predict(self.spectral_embeddings)

        log_memory_usage("After spectral clustering")
        logging.info("Spectral clustering on stratified subset completed")
        logging.info(f"Spectral embedding shape: {self.spectral_embeddings.shape}")

        return self.cluster_labels_stratified

    def compute_cluster_centroids(self):
        """Compute cluster centroids in spectral space."""
        logging.info("Computing cluster centroids in spectral space...")

        self.cluster_centroids = np.zeros((self.n_clusters, self.spectral_embeddings.shape[1]))

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_stratified == cluster_id
            cluster_points = self.spectral_embeddings[cluster_mask]
            self.cluster_centroids[cluster_id] = cluster_points.mean(axis=0)

        logging.info(f"Computed {self.n_clusters} cluster centroids")
        logging.info(f"Centroid shape: {self.cluster_centroids.shape}")

    def project_to_spectral_space(self,
                                  remaining_embeddings: np.ndarray,
                                  stratified_embeddings: np.ndarray,
                                  k: int = 3) -> np.ndarray:
        """
        Project remaining nodes into spectral space using k-NN weighted average.

        Args:
            remaining_embeddings: Embeddings of remaining nodes (original space)
            stratified_embeddings: Embeddings of stratified nodes (original space)
            k: Number of nearest neighbors to use for projection

        Returns:
            Spectral embeddings for remaining nodes
        """
        logging.info(f"Projecting {len(remaining_embeddings)} remaining nodes to spectral space using k={k} neighbors")
        log_memory_usage("Before projection")

        # Build k-NN index on stratified subset
        nbrs = NearestNeighbors(n_neighbors=k, metric='cosine', n_jobs=-1, algorithm='auto')
        nbrs.fit(stratified_embeddings)

        # Find k nearest neighbors for each remaining node
        distances, indices = nbrs.kneighbors(remaining_embeddings)

        # Convert distances to similarities
        similarities = 1 - distances

        # Avoid division by zero
        similarities = np.maximum(similarities, 1e-10)

        # Project to spectral space using weighted average
        remaining_spectral = np.zeros((len(remaining_embeddings), self.spectral_embeddings.shape[1]))

        for i in range(len(remaining_embeddings)):
            # Get neighbor indices and their weights
            neighbor_indices = indices[i]
            weights = similarities[i]

            # Normalize weights
            weights = weights / weights.sum()

            # Weighted average of neighbor spectral embeddings
            for j, neighbor_idx in enumerate(neighbor_indices):
                remaining_spectral[i] += weights[j] * self.spectral_embeddings[neighbor_idx]

        # Clean up
        del nbrs, distances, indices, similarities
        gc.collect()

        log_memory_usage("After projection")
        logging.info("Projection to spectral space completed")

        return remaining_spectral

    def assign_to_clusters(self, remaining_spectral: np.ndarray) -> np.ndarray:
        """
        Assign remaining nodes to clusters based on nearest centroid (cosine similarity).

        Args:
            remaining_spectral: Spectral embeddings of remaining nodes

        Returns:
            Cluster labels for remaining nodes
        """
        logging.info("Assigning remaining nodes to clusters...")
        log_memory_usage("Before assignment")

        # Compute cosine similarity between remaining nodes and centroids
        similarities = cosine_similarity(remaining_spectral, self.cluster_centroids)

        # Assign to cluster with highest similarity
        cluster_labels_remaining = np.argmax(similarities, axis=1)

        log_memory_usage("After assignment")
        logging.info(f"Assigned {len(cluster_labels_remaining)} nodes to clusters")

        return cluster_labels_remaining


class MiniBatchSpectralClusteringPipeline:
    """Complete mini-batch spectral clustering pipeline."""

    def __init__(self,
                 embeddings_file: str,
                 n_clusters: int = 40,
                 graph_k: int = 20,
                 projection_k: int = 3,
                 stratified_ratio: float = 0.1,
                 graph_type: str = "knn",
                 output_dir: str = "results/minibatch_spectral_clustering",
                 random_state: int = 42):
        """
        Initialize mini-batch clustering pipeline.

        Args:
            embeddings_file: Path to embeddings .npy file with labels
            n_clusters: Number of clusters
            graph_k: Number of neighbors for k-NN graph construction on stratified subset
            projection_k: Number of neighbors for projecting remaining nodes
            stratified_ratio: Ratio of stratified sample (e.g., 0.1 for 10%)
            graph_type: Type of similarity graph ('knn' only for now)
            output_dir: Output directory for results
            random_state: Random state for reproducibility
        """
        self.embeddings_file = embeddings_file
        self.n_clusters = n_clusters
        self.graph_k = graph_k
        self.projection_k = projection_k
        self.stratified_ratio = stratified_ratio
        self.graph_type = graph_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Components
        self.embeddings = None
        self.node_ids = None
        self.class_labels = None

        self.stratified_embeddings = None
        self.stratified_ids = None
        self.stratified_labels = None

        self.remaining_embeddings = None
        self.remaining_ids = None
        self.remaining_labels = None

        self.sampler = StratifiedSampler(stratified_ratio, random_state)
        self.similarity_graph = None
        self.clustering = None

        self.final_cluster_labels = None  # Combined labels for all nodes

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"minibatch_clustering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

        log_memory_usage("After loading embeddings")

    def create_stratified_sample(self):
        """Create stratified sample."""
        (self.stratified_embeddings, self.stratified_ids, self.stratified_labels,
         self.remaining_embeddings, self.remaining_ids, self.remaining_labels) = \
            self.sampler.create_stratified_sample(
                self.embeddings, self.node_ids, self.class_labels
            )

    def run_clustering(self) -> Dict[str, Any]:
        """Run complete mini-batch clustering pipeline."""
        logging.info("Starting mini-batch spectral clustering pipeline...")
        log_memory_usage("Pipeline start")

        # Step 1: Load embeddings
        self.load_embeddings_with_labels()

        # Step 2: Create stratified sample
        self.create_stratified_sample()

        # Step 3: Build similarity graph on stratified subset
        logging.info(f"Building {self.graph_type} graph on stratified subset with k={self.graph_k}")
        self.similarity_graph = MiniBatchSimilarityGraph(self.stratified_embeddings, self.stratified_ids)
        affinity_matrix = self.similarity_graph.get_knn_similarity_graph(k=self.graph_k)

        # Step 4: Run spectral clustering on stratified subset
        self.clustering = MiniBatchSpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        stratified_cluster_labels = self.clustering.fit_stratified(affinity_matrix)

        # Clean up affinity matrix
        del affinity_matrix
        gc.collect()

        # Step 5: Compute cluster centroids in spectral space
        self.clustering.compute_cluster_centroids()

        # Step 6: Project remaining nodes to spectral space
        remaining_spectral = self.clustering.project_to_spectral_space(
            self.remaining_embeddings,
            self.stratified_embeddings,
            k=self.projection_k
        )

        # Step 7: Assign remaining nodes to clusters
        remaining_cluster_labels = self.clustering.assign_to_clusters(remaining_spectral)

        # Step 8: Combine labels - reconstruct full label array in original order
        self.final_cluster_labels = self._reconstruct_full_labels(
            stratified_cluster_labels, remaining_cluster_labels
        )

        # Step 9: Evaluate on complete dataset
        metrics = self.evaluate_clustering()

        # Step 10: Analyze cluster composition
        composition_analysis = self.analyze_cluster_composition()

        # Step 11: Visualize results
        self.visualize_results()

        # Step 12: Prepare results
        results = {
            'method': 'minibatch_spectral_clustering',
            'embeddings_file': str(self.embeddings_file),
            'stratified_ratio': self.stratified_ratio,
            'graph_k': self.graph_k,
            'projection_k': self.projection_k,
            'n_clusters': self.n_clusters,
            'n_papers_total': len(self.embeddings),
            'n_papers_stratified': len(self.stratified_embeddings),
            'n_papers_remaining': len(self.remaining_embeddings),
            'embedding_dim': self.embeddings.shape[1],
            'evaluation_metrics': metrics,
            'composition_analysis': composition_analysis,
            'timestamp': datetime.now().isoformat()
        }

        # Save results to JSON
        results_file = self.output_dir / "clustering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_file}")

        # Print formatted results
        self.print_results_formatted(results)

        log_memory_usage("Pipeline end")
        logging.info("Mini-batch clustering pipeline completed successfully!")

        return results

    def _reconstruct_full_labels(self,
                                 stratified_labels: np.ndarray,
                                 remaining_labels: np.ndarray) -> np.ndarray:
        """
        Reconstruct full label array in original order.

        Args:
            stratified_labels: Cluster labels for stratified subset
            remaining_labels: Cluster labels for remaining nodes

        Returns:
            Full cluster labels array in original order
        """
        logging.info("Reconstructing full label array in original order...")

        # Create a dictionary mapping node_id to cluster_label
        label_dict = {}

        # Add stratified labels
        for node_id, label in zip(self.stratified_ids, stratified_labels):
            label_dict[node_id] = label

        # Add remaining labels
        for node_id, label in zip(self.remaining_ids, remaining_labels):
            label_dict[node_id] = label

        # Reconstruct in original order
        full_labels = np.array([label_dict[node_id] for node_id in self.node_ids])

        logging.info(f"Reconstructed full labels array with {len(full_labels)} labels")

        return full_labels

    def evaluate_clustering(self) -> Dict[str, float]:
        """
        Evaluate clustering using ARI, NMI, and V-measure on complete dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Evaluating clustering on complete dataset...")
        log_memory_usage("Before evaluation")

        metrics = {}

        # Adjusted Rand Index
        ari = adjusted_rand_score(self.class_labels, self.final_cluster_labels)
        metrics['adjusted_rand_index'] = float(ari)
        logging.info(f"Adjusted Rand Index: {ari:.4f}")

        # Normalized Mutual Information
        nmi = normalized_mutual_info_score(self.class_labels, self.final_cluster_labels)
        metrics['normalized_mutual_info'] = float(nmi)
        logging.info(f"Normalized Mutual Information: {nmi:.4f}")

        # V-measure
        v_measure = v_measure_score(self.class_labels, self.final_cluster_labels)
        metrics['v_measure'] = float(v_measure)
        logging.info(f"V-measure: {v_measure:.4f}")

        log_memory_usage("After evaluation")

        return metrics

    def analyze_cluster_composition(self) -> Dict[str, Any]:
        """
        Analyze the composition of each cluster - which class labels are present.

        Returns:
            Dictionary with detailed cluster composition analysis
        """
        logging.info("Analyzing cluster composition...")

        cluster_composition = {}
        unique_clusters = np.unique(self.final_cluster_labels)
        unique_classes = np.unique(self.class_labels)

        # Create detailed composition for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = self.final_cluster_labels == cluster_id
            cluster_ground_truth = self.class_labels[cluster_mask]

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
            csv_path = self.output_dir / "cluster_composition_detailed.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Detailed cluster composition saved to {csv_path}")
            print(f"Detailed cluster composition saved to {csv_path}")

        except Exception as e:
            logging.warning(f"Could not save cluster composition to CSV: {e}")

    def visualize_results(self):
        """Create visualizations for clustering results."""
        logging.info("Creating visualizations...")

        # 1. Distribution comparison
        self._plot_distribution_comparison()

        # 2. Confusion matrix
        self._plot_confusion_matrix()

        # 3. Cluster sizes
        self._plot_cluster_sizes()

        logging.info("Visualizations completed")

    def _plot_distribution_comparison(self):
        """Plot ground truth vs predicted distribution."""
        log_memory_usage("Before distribution plot")

        gt_unique, gt_counts = np.unique(self.class_labels, return_counts=True)
        pred_unique, pred_counts = np.unique(self.final_cluster_labels, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Ground truth distribution
        bars1 = ax1.bar(gt_unique, gt_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Ground Truth Class Distribution')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars1, gt_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(gt_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        # Predicted cluster distribution
        bars2 = ax2.bar(pred_unique, pred_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_title('Predicted Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Papers')
        ax2.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars2, pred_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pred_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.suptitle('Ground Truth vs Predicted Distribution (Mini-batch)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / "distribution_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Distribution comparison saved to {save_path}")
        plt.close()

        del fig, ax1, ax2, bars1, bars2
        gc.collect()
        log_memory_usage("After distribution plot")

    def _plot_confusion_matrix(self):
        """Plot confusion matrix."""
        log_memory_usage("Before confusion matrix")

        df_confusion = pd.crosstab(self.class_labels, self.final_cluster_labels, margins=True)

        plt.figure(figsize=(12, 10))
        confusion_subset = df_confusion.iloc[:-1, :-1]
        sns.heatmap(confusion_subset, annot=False, cmap='Blues', fmt='d')

        plt.title('Confusion Matrix: Ground Truth vs Predicted Clusters (Mini-batch)')
        plt.xlabel('Predicted Cluster')
        plt.ylabel('Ground Truth Class')

        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {save_path}")
        plt.close()

        del df_confusion, confusion_subset
        gc.collect()
        log_memory_usage("After confusion matrix")

    def _plot_cluster_sizes(self):
        """Plot cluster size distribution."""
        unique_labels, counts = np.unique(self.final_cluster_labels, return_counts=True)

        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts, alpha=0.7)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Papers')
        plt.title('Cluster Size Distribution (Mini-batch)')
        plt.xticks(unique_labels)

        for i, v in enumerate(counts):
            plt.text(unique_labels[i], v + 0.5, str(v), ha='center')

        save_path = self.output_dir / "cluster_sizes.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Cluster sizes saved to {save_path}")
        plt.close()

    def print_results_formatted(self, results: dict):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("MINI-BATCH SPECTRAL CLUSTERING RESULTS")
        print("=" * 80)
        print(f"Method: {results['method']}")
        print(f"Stratified ratio: {results['stratified_ratio']:.1%}")
        print(f"Graph construction k: {results['graph_k']}")
        print(f"Projection k: {results['projection_k']}")
        print(f"Total papers: {results['n_papers_total']}")
        print(f"Stratified papers: {results['n_papers_stratified']}")
        print(f"Remaining papers: {results['n_papers_remaining']}")
        print(f"Number of clusters: {results['n_clusters']}")

        print(f"\nEVALUATION METRICS (Complete Dataset):")
        for metric, value in results['evaluation_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nCOMPOSITION SUMMARY:")
        composition = results['composition_analysis']['summary_statistics']
        print(f"  Clusters with single class: {composition['clusters_with_single_class']}")
        print(f"  Clusters with multiple classes: {composition['clusters_with_multiple_classes']}")
        print(f"  Average classes per cluster: {composition['avg_classes_per_cluster']:.2f}")
        print(f"  Average dominant class percentage: {composition['avg_dominant_class_percentage']:.1f}%")

        print("=" * 80 + "\n")


def main():
    """Main function to run mini-batch spectral clustering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run mini-batch spectral clustering with stratified sampling"
    )
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
        help="Number of clusters (default: 40)"
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=20,
        help="Number of neighbors for k-NN graph construction on stratified subset (default: 20)"
    )
    parser.add_argument(
        "--projection-k",
        type=int,
        default=3,
        help="Number of neighbors for projecting remaining nodes (default: 3)"
    )
    parser.add_argument(
        "--stratified-ratio",
        type=float,
        default=0.1,
        help="Ratio of stratified sample (default: 0.1 for 10%%)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/minibatch_spectral_clustering",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Create and run pipeline
    pipeline = MiniBatchSpectralClusteringPipeline(
        embeddings_file=args.embeddings_file,
        n_clusters=args.n_clusters,
        graph_k=args.graph_k,
        projection_k=args.projection_k,
        stratified_ratio=args.stratified_ratio,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    results = pipeline.run_clustering()

    # Print summary
    print("\n" + "=" * 80)
    print("MINI-BATCH SPECTRAL CLUSTERING COMPLETED")
    print("=" * 80)
    print(f"ARI: {results['evaluation_metrics']['adjusted_rand_index']:.4f}")
    print(f"NMI: {results['evaluation_metrics']['normalized_mutual_info']:.4f}")
    print(f"V-measure: {results['evaluation_metrics']['v_measure']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()