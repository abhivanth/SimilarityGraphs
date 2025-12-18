import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
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
from scipy.sparse import csr_matrix
from collections import defaultdict


def log_memory_usage(step_name: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {step_name}: {memory_mb:.1f} MB")


class StratifiedCitationSampler:
    """Create stratified sample of nodes preserving class distribution."""

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
                                 nodes_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create stratified sample and return both stratified and remaining sets.

        Args:
            nodes_df: DataFrame with columns ['node_id', 'class_idx', ...]

        Returns:
            Tuple of (stratified_df, remaining_df)
        """
        logging.info(f"Creating stratified sample with ratio={self.stratified_ratio}")
        log_memory_usage("Before stratified sampling")

        # Perform stratified split
        stratified_df, remaining_df = train_test_split(
            nodes_df,
            test_size=(1 - self.stratified_ratio),
            stratify=nodes_df['class_idx'],
            random_state=self.random_state
        )

        logging.info(f"Stratified sample size: {len(stratified_df)}")
        logging.info(f"Remaining sample size: {len(remaining_df)}")

        # Verify class distribution preservation
        original_dist = dict(zip(*np.unique(nodes_df['class_idx'], return_counts=True)))
        stratified_dist = dict(zip(*np.unique(stratified_df['class_idx'], return_counts=True)))

        logging.info("Original class distribution (counts): %s", original_dist)
        logging.info("Stratified class distribution (counts): %s", stratified_dist)

        log_memory_usage("After stratified sampling")

        return stratified_df, remaining_df


class CitationGraph:
    """Build and manage citation graph for spectral clustering."""

    def __init__(self, edges_df: pd.DataFrame):
        """
        Initialize citation graph.

        Args:
            edges_df: DataFrame with columns ['source', 'target']
        """
        self.edges_df = edges_df
        self.adjacency_matrix = None
        self.node_to_idx = None
        self.idx_to_node = None

    def build_induced_subgraph(self, node_ids: List[int]) -> csr_matrix:
        """
        Build induced subgraph containing only edges within specified nodes.
        Makes the graph undirected/symmetric for spectral clustering.

        Args:
            node_ids: List of node IDs to include in subgraph

        Returns:
            Sparse adjacency matrix (symmetric)
        """
        logging.info(f"Building induced subgraph for {len(node_ids)} nodes...")
        log_memory_usage("Before subgraph construction")

        node_set = set(node_ids)

        # Filter edges to only include nodes in the subset
        valid_edges = self.edges_df[
            (self.edges_df['source'].isin(node_set)) &
            (self.edges_df['target'].isin(node_set))
            ]

        logging.info(f"Valid edges within subgraph: {len(valid_edges)}")

        # Create node mapping
        node_list = sorted(node_set)
        self.node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # Create adjacency matrix (symmetric for undirected graph)
        n_nodes = len(node_list)
        row_indices = []
        col_indices = []

        for _, edge in valid_edges.iterrows():
            source_idx = self.node_to_idx[edge['source']]
            target_idx = self.node_to_idx[edge['target']]

            # Add edge in both directions (undirected)
            if source_idx != target_idx:
                row_indices.extend([source_idx, target_idx])
                col_indices.extend([target_idx, source_idx])

        # Create sparse matrix with binary edges
        data = np.ones(len(row_indices))
        self.adjacency_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes)
        )

        logging.info(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        logging.info(f"Number of edges (symmetric): {self.adjacency_matrix.nnz // 2}")
        logging.info(f"Graph density: {self.adjacency_matrix.nnz / (n_nodes * n_nodes):.6f}")

        # Check graph connectivity
        avg_degree = self.adjacency_matrix.nnz / n_nodes
        logging.info(f"Average node degree: {avg_degree:.2f}")

        if avg_degree < 2:
            logging.warning(f"WARNING: Graph is very sparse (avg degree: {avg_degree:.2f}). "
                            f"This may affect clustering quality and convergence.")
            logging.warning("Consider: (1) Using larger stratified_ratio, or (2) Different clustering approach")

        log_memory_usage("After subgraph construction")

        return self.adjacency_matrix


class CitationVectorizer:
    """Create citation vectors for computing citation similarity."""

    def __init__(self, edges_df: pd.DataFrame, all_node_ids: List[int]):
        """
        Initialize citation vectorizer.

        Args:
            edges_df: DataFrame with columns ['source', 'target']
            all_node_ids: List of all node IDs in dataset
        """
        self.edges_df = edges_df
        self.all_node_ids = sorted(all_node_ids)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.all_node_ids)}

        # Build citation dictionaries
        self._build_citation_dicts()

    def _build_citation_dicts(self):
        """Build dictionaries of incoming and outgoing citations."""
        logging.info("Building citation dictionaries...")

        self.out_citations = defaultdict(set)  # Papers this node cites
        self.in_citations = defaultdict(set)  # Papers that cite this node

        for _, edge in self.edges_df.iterrows():
            source = edge['source']
            target = edge['target']

            if source in self.node_to_idx and target in self.node_to_idx:
                self.out_citations[source].add(target)
                self.in_citations[target].add(source)

        logging.info(f"Citation dictionaries built for {len(self.all_node_ids)} nodes")

    def get_citation_vector(self, node_id: int) -> np.ndarray:
        """
        Get citation vector for a node (in + out citations).

        Args:
            node_id: Node ID

        Returns:
            Binary vector of length n_nodes
        """
        vector = np.zeros(len(self.all_node_ids))

        # Mark outgoing citations
        for cited_node in self.out_citations.get(node_id, []):
            if cited_node in self.node_to_idx:
                vector[self.node_to_idx[cited_node]] = 1

        # Mark incoming citations
        for citing_node in self.in_citations.get(node_id, []):
            if citing_node in self.node_to_idx:
                vector[self.node_to_idx[citing_node]] = 1

        return vector

    def compute_citation_similarity(self,
                                    remaining_nodes: List[int],
                                    stratified_nodes: List[int]) -> np.ndarray:
        """
        Compute cosine similarity between remaining and stratified nodes based on citations.

        Args:
            remaining_nodes: List of remaining node IDs
            stratified_nodes: List of stratified node IDs

        Returns:
            Similarity matrix (n_remaining x n_stratified)
        """
        logging.info(f"Computing citation similarity for {len(remaining_nodes)} remaining nodes...")
        log_memory_usage("Before citation similarity")

        # Build citation vectors
        remaining_vectors = np.array([self.get_citation_vector(node) for node in remaining_nodes])
        stratified_vectors = np.array([self.get_citation_vector(node) for node in stratified_nodes])

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(remaining_vectors, stratified_vectors)

        log_memory_usage("After citation similarity")
        logging.info("Citation similarity computation completed")

        return similarity_matrix


class MiniBatchCitationSpectralClustering:
    """Spectral clustering for citation networks with out-of-sample extension."""

    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Initialize mini-batch citation spectral clustering.

        Args:
            n_clusters: Number of clusters to find
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clustering_model = None
        self.cluster_labels_stratified = None
        self.spectral_embeddings = None
        self.cluster_centroids = None

        logging.info(f"Initialized mini-batch citation spectral clustering with {n_clusters} clusters")

    def fit_stratified(self, adjacency_matrix: csr_matrix) -> np.ndarray:
        """
        Fit spectral clustering on stratified subset and get spectral embeddings.

        Args:
            adjacency_matrix: Precomputed sparse adjacency matrix for stratified subset

        Returns:
            Cluster labels for stratified subset
        """
        logging.info("Fitting spectral clustering on stratified citation subgraph...")
        log_memory_usage("Before spectral clustering")

        # Aggressive memory cleanup
        gc.collect()

        # Compute spectral embeddings explicitly
        from sklearn.cluster import KMeans
        from scipy.sparse.linalg import eigsh
        from sklearn.preprocessing import normalize

        logging.info("Computing spectral embeddings...")

        # Compute normalized Laplacian
        D = np.array(adjacency_matrix.sum(axis=1)).flatten()
        D[D == 0] = 1  # Avoid division by zero
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(D))
        L_norm = sparse.eye(adjacency_matrix.shape[0]) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt

        # Compute eigenvectors with better convergence parameters
        try:
            # Try with default solver first
            eigenvalues, eigenvectors = eigsh(
                L_norm,
                k=self.n_clusters,
                which='SM',
                maxiter=adjacency_matrix.shape[0] * 10,  # Increase iterations
                tol=1e-3  # Relax tolerance
            )
        except Exception as e:
            logging.warning(f"ARPACK eigsh failed: {e}")
            logging.info("Falling back to dense eigendecomposition...")

            # Convert to dense for very sparse graphs
            L_norm_dense = L_norm.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_norm_dense)

            # Take smallest k eigenvalues
            idx = eigenvalues.argsort()[:self.n_clusters]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Normalize eigenvectors
        self.spectral_embeddings = normalize(eigenvectors)

        # Run k-means on spectral embeddings
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
            if len(cluster_points) > 0:
                self.cluster_centroids[cluster_id] = cluster_points.mean(axis=0)

        logging.info(f"Computed {self.n_clusters} cluster centroids")
        logging.info(f"Centroid shape: {self.cluster_centroids.shape}")

    def project_to_spectral_space(self,
                                  similarity_matrix: np.ndarray,
                                  k: int = 3) -> np.ndarray:
        """
        Project remaining nodes into spectral space using k-NN weighted average.
        Uses citation similarity to find neighbors.

        Args:
            similarity_matrix: Citation similarity between remaining and stratified nodes
            k: Number of nearest neighbors to use for projection

        Returns:
            Spectral embeddings for remaining nodes
        """
        logging.info(f"Projecting {similarity_matrix.shape[0]} remaining nodes to spectral space using k={k} neighbors")
        log_memory_usage("Before projection")

        n_remaining = similarity_matrix.shape[0]
        remaining_spectral = np.zeros((n_remaining, self.spectral_embeddings.shape[1]))

        for i in range(n_remaining):
            # Get similarities to all stratified nodes
            similarities = similarity_matrix[i]

            # Find k nearest neighbors (relaxed threshold - take top k even if low similarity)
            if np.sum(similarities > 0) >= k:
                # Enough neighbors with positive similarity
                top_k_indices = np.argpartition(similarities, -k)[-k:]
            else:
                # Relaxed: take all positive similarities, or top k if none
                positive_sim_indices = np.where(similarities > 0)[0]
                if len(positive_sim_indices) > 0:
                    top_k_indices = positive_sim_indices
                else:
                    # No positive similarities - take top k anyway
                    top_k_indices = np.argpartition(similarities, -k)[-k:]

            # Get weights (similarities)
            weights = similarities[top_k_indices]

            # Normalize weights (avoid division by zero)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Equal weights if all similarities are 0
                weights = np.ones(len(top_k_indices)) / len(top_k_indices)

            # Weighted average of neighbor spectral embeddings
            for j, neighbor_idx in enumerate(top_k_indices):
                remaining_spectral[i] += weights[j] * self.spectral_embeddings[neighbor_idx]

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


class MiniBatchCitationSpectralClusteringPipeline:
    """Complete mini-batch spectral clustering pipeline for citation networks."""

    def __init__(self,
                 nodes_csv: str,
                 edges_csv: str,
                 n_clusters: int = 40,
                 projection_k: int = 3,
                 stratified_ratio: float = 0.1,
                 output_dir: str = "results/citation_minibatch_spectral_clustering",
                 random_state: int = 42):
        """
        Initialize mini-batch citation clustering pipeline.

        Args:
            nodes_csv: Path to nodes CSV file (columns: node_id, class_idx)
            edges_csv: Path to edges CSV file (columns: source, target)
            n_clusters: Number of clusters
            projection_k: Number of neighbors for projecting remaining nodes
            stratified_ratio: Ratio of stratified sample (e.g., 0.1 for 10%)
            output_dir: Output directory for results
            random_state: Random state for reproducibility
        """
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.n_clusters = n_clusters
        self.projection_k = projection_k
        self.stratified_ratio = stratified_ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Data
        self.nodes_df = None
        self.edges_df = None

        self.stratified_df = None
        self.remaining_df = None

        # Components
        self.sampler = StratifiedCitationSampler(stratified_ratio, random_state)
        self.citation_graph = None
        self.citation_vectorizer = None
        self.clustering = None

        self.final_cluster_labels = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"citation_minibatch_clustering_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_data(self):
        """Load nodes and edges CSV files."""
        logging.info(f"Loading citation network data...")
        log_memory_usage("Before loading data")

        # Load nodes
        self.nodes_df = pd.read_csv(self.nodes_csv)
        logging.info(f"Loaded {len(self.nodes_df)} nodes")
        logging.info(f"Nodes columns: {self.nodes_df.columns.tolist()}")

        # Load edges
        self.edges_df = pd.read_csv(self.edges_csv)
        logging.info(f"Loaded {len(self.edges_df)} edges")
        logging.info(f"Edges columns: {self.edges_df.columns.tolist()}")

        # Statistics
        logging.info(f"Number of unique classes: {self.nodes_df['class_idx'].nunique()}")

        log_memory_usage("After loading data")

    def create_stratified_sample(self):
        """Create stratified sample."""
        self.stratified_df, self.remaining_df = self.sampler.create_stratified_sample(self.nodes_df)

    def run_clustering(self) -> Dict[str, Any]:
        """Run complete mini-batch citation clustering pipeline."""
        logging.info("Starting mini-batch citation spectral clustering pipeline...")
        log_memory_usage("Pipeline start")

        # Step 1: Load data
        self.load_data()

        # Step 2: Create stratified sample
        self.create_stratified_sample()

        # Step 3: Build induced subgraph for stratified subset
        logging.info("Building citation graph for stratified subset...")
        self.citation_graph = CitationGraph(self.edges_df)
        stratified_node_ids = self.stratified_df['node_id'].tolist()
        adjacency_matrix = self.citation_graph.build_induced_subgraph(stratified_node_ids)

        # Step 4: Run spectral clustering on stratified subset
        self.clustering = MiniBatchCitationSpectralClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        stratified_cluster_labels = self.clustering.fit_stratified(adjacency_matrix)

        # Clean up adjacency matrix
        del adjacency_matrix
        gc.collect()

        # Step 5: Compute cluster centroids in spectral space
        self.clustering.compute_cluster_centroids()

        # Step 6: Build citation vectorizer for similarity computation
        logging.info("Building citation vectorizer...")
        all_node_ids = self.nodes_df['node_id'].tolist()
        self.citation_vectorizer = CitationVectorizer(self.edges_df, all_node_ids)

        # Step 7: Compute citation similarity between remaining and stratified nodes
        remaining_node_ids = self.remaining_df['node_id'].tolist()
        similarity_matrix = self.citation_vectorizer.compute_citation_similarity(
            remaining_node_ids, stratified_node_ids
        )

        # Step 8: Project remaining nodes to spectral space
        remaining_spectral = self.clustering.project_to_spectral_space(
            similarity_matrix,
            k=self.projection_k
        )

        # Clean up similarity matrix
        del similarity_matrix
        gc.collect()

        # Step 9: Assign remaining nodes to clusters
        remaining_cluster_labels = self.clustering.assign_to_clusters(remaining_spectral)

        # Step 10: Combine labels
        self.final_cluster_labels = self._reconstruct_full_labels(
            stratified_cluster_labels, remaining_cluster_labels
        )

        # Step 11: Evaluate on complete dataset
        metrics = self.evaluate_clustering()

        # Step 12: Analyze cluster composition
        composition_analysis = self.analyze_cluster_composition()

        # Step 13: Visualize results
        self.visualize_results()

        # Step 14: Prepare results
        results = {
            'method': 'minibatch_citation_spectral_clustering',
            'nodes_csv': str(self.nodes_csv),
            'edges_csv': str(self.edges_csv),
            'stratified_ratio': self.stratified_ratio,
            'projection_k': self.projection_k,
            'n_clusters': self.n_clusters,
            'n_papers_total': len(self.nodes_df),
            'n_papers_stratified': len(self.stratified_df),
            'n_papers_remaining': len(self.remaining_df),
            'n_edges_total': len(self.edges_df),
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
        logging.info("Mini-batch citation clustering pipeline completed successfully!")

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

        # Create mapping from node_id to cluster_label
        label_dict = {}

        # Add stratified labels
        for node_id, label in zip(self.stratified_df['node_id'], stratified_labels):
            label_dict[node_id] = label

        # Add remaining labels
        for node_id, label in zip(self.remaining_df['node_id'], remaining_labels):
            label_dict[node_id] = label

        # Reconstruct in original order
        full_labels = np.array([label_dict[node_id] for node_id in self.nodes_df['node_id']])

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

        ground_truth = self.nodes_df['class_idx'].values

        metrics = {}

        # Adjusted Rand Index
        ari = adjusted_rand_score(ground_truth, self.final_cluster_labels)
        metrics['adjusted_rand_index'] = float(ari)
        logging.info(f"Adjusted Rand Index: {ari:.4f}")

        # Normalized Mutual Information
        nmi = normalized_mutual_info_score(ground_truth, self.final_cluster_labels)
        metrics['normalized_mutual_info'] = float(nmi)
        logging.info(f"Normalized Mutual Information: {nmi:.4f}")

        # V-measure
        v_measure = v_measure_score(ground_truth, self.final_cluster_labels)
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
        unique_classes = np.unique(self.nodes_df['class_idx'])

        # Create detailed composition for each cluster
        for cluster_id in unique_clusters:
            cluster_mask = self.final_cluster_labels == cluster_id
            cluster_ground_truth = self.nodes_df['class_idx'].values[cluster_mask]

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

        # Save to CSV
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

            # Show top 3 classes
            sorted_classes = sorted(comp['class_counts'].items(), key=lambda x: x[1], reverse=True)
            print("  Class distribution:")
            for class_id, count in sorted_classes[:3]:
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
            csv_path = self.output_dir / "citation_cluster_composition_detailed.csv"
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

        ground_truth = self.nodes_df['class_idx'].values
        gt_unique, gt_counts = np.unique(ground_truth, return_counts=True)
        pred_unique, pred_counts = np.unique(self.final_cluster_labels, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Ground truth
        bars1 = ax1.bar(gt_unique, gt_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_title('Ground Truth Class Distribution')
        ax1.set_xlabel('Class ID')
        ax1.set_ylabel('Number of Papers')
        ax1.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars1, gt_counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(gt_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        # Predicted
        bars2 = ax2.bar(pred_unique, pred_counts, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax2.set_title('Predicted Cluster Distribution')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Papers')
        ax2.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars2, pred_counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pred_counts) * 0.01,
                     str(count), ha='center', va='bottom', fontsize=9)

        plt.suptitle('Ground Truth vs Predicted Distribution (Citation MiniBatch)', fontsize=14, fontweight='bold')
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

        ground_truth = self.nodes_df['class_idx'].values
        df_confusion = pd.crosstab(ground_truth, self.final_cluster_labels, margins=True)

        plt.figure(figsize=(12, 10))
        confusion_subset = df_confusion.iloc[:-1, :-1]
        sns.heatmap(confusion_subset, annot=False, cmap='Blues', fmt='d')

        plt.title('Confusion Matrix: Ground Truth vs Predicted Clusters (Citation MiniBatch)')
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
        plt.title('Cluster Size Distribution (Citation MiniBatch)')
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
        print("MINI-BATCH CITATION SPECTRAL CLUSTERING RESULTS")
        print("=" * 80)
        print(f"Method: {results['method']}")
        print(f"Stratified ratio: {results['stratified_ratio']:.1%}")
        print(f"Projection k: {results['projection_k']}")
        print(f"Total papers: {results['n_papers_total']}")
        print(f"Stratified papers: {results['n_papers_stratified']}")
        print(f"Remaining papers: {results['n_papers_remaining']}")
        print(f"Total edges: {results['n_edges_total']}")
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
    """Main function to run mini-batch citation spectral clustering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run mini-batch spectral clustering on citation networks"
    )
    parser.add_argument(
        "--nodes-csv",
        type=str,
        required=True,
        help="Path to nodes CSV file (columns: node_id, class_idx)"
    )
    parser.add_argument(
        "--edges-csv",
        type=str,
        required=True,
        help="Path to edges CSV file (columns: source, target)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=40,
        help="Number of clusters (default: 40)"
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
        default="results/citation_minibatch_spectral_clustering",
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
    pipeline = MiniBatchCitationSpectralClusteringPipeline(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        n_clusters=args.n_clusters,
        projection_k=args.projection_k,
        stratified_ratio=args.stratified_ratio,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    results = pipeline.run_clustering()

    # Print summary
    print("\n" + "=" * 80)
    print("MINI-BATCH CITATION SPECTRAL CLUSTERING COMPLETED")
    print("=" * 80)
    print(f"ARI: {results['evaluation_metrics']['adjusted_rand_index']:.4f}")
    print(f"NMI: {results['evaluation_metrics']['normalized_mutual_info']:.4f}")
    print(f"V-measure: {results['evaluation_metrics']['v_measure']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()