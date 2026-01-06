import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
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
from scipy.sparse.linalg import eigsh
from collections import defaultdict

try:
    from mvlearn.cluster import MultiviewSpectralClustering

    MVLEARN_AVAILABLE = True
except ImportError:
    MVLEARN_AVAILABLE = False
    logging.warning("mvlearn not available. Will use fallback concatenation method.")


def log_memory_usage(step_name: str):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logging.info(f"[MEMORY] {step_name}: {memory_mb:.1f} MB")


class StratifiedSampler:
    """Create stratified sample preserving class distribution with optional neighborhood expansion."""

    def __init__(self, stratified_ratio: float = 0.1, include_neighbors: bool = False, random_state: int = 42):
        """
        Initialize stratified sampler.

        Args:
            stratified_ratio: Ratio of data to sample (e.g., 0.1 for 10%)
            include_neighbors: Whether to include 1-hop neighbors of seed nodes
            random_state: Random state for reproducibility
        """
        self.stratified_ratio = stratified_ratio
        self.include_neighbors = include_neighbors
        self.random_state = random_state

    def create_stratified_sample(self,
                                 embeddings: np.ndarray,
                                 node_ids: List[int],
                                 class_labels: np.ndarray,
                                 edges_df: pd.DataFrame = None) -> Tuple[np.ndarray, List[int], np.ndarray,
    np.ndarray, List[int], np.ndarray]:
        """
        Create stratified sample with optional neighborhood expansion.

        Args:
            embeddings: Full embedding matrix
            node_ids: List of node IDs
            class_labels: Ground truth class labels
            edges_df: Optional DataFrame with edges for neighborhood expansion

        Returns:
            Tuple of (stratified_embeddings, stratified_ids, stratified_labels,
                     remaining_embeddings, remaining_ids, remaining_labels)
        """
        logging.info(f"Creating stratified sample with ratio={self.stratified_ratio}, "
                     f"include_neighbors={self.include_neighbors}")
        log_memory_usage("Before stratified sampling")

        # Step 1: Create seed nodes via stratified split
        indices = np.arange(len(embeddings))
        seed_indices, remaining_indices = train_test_split(
            indices,
            test_size=(1 - self.stratified_ratio),
            stratify=class_labels,
            random_state=self.random_state
        )

        seed_ids = [node_ids[i] for i in seed_indices]
        logging.info(f"Seed sample size: {len(seed_ids)} ({self.stratified_ratio:.1%})")

        # Step 2: Expand with neighbors if enabled
        if self.include_neighbors and edges_df is not None:
            logging.info("Performing neighborhood expansion (1-hop)...")

            seed_node_ids_set = set(seed_ids)

            # Find all 1-hop neighbors
            neighbors_out = set(edges_df[edges_df['source'].isin(seed_node_ids_set)]['target'].tolist())
            neighbors_in = set(edges_df[edges_df['target'].isin(seed_node_ids_set)]['source'].tolist())
            all_neighbors = (neighbors_out | neighbors_in) - seed_node_ids_set

            logging.info(f"Found {len(all_neighbors)} neighbor nodes (1-hop)")

            # Map node IDs to indices
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            # Get indices of neighbors that are in our dataset
            neighbor_indices = [node_id_to_idx[nid] for nid in all_neighbors if nid in node_id_to_idx]

            if neighbor_indices:
                # Combine seed + neighbor indices
                stratified_indices = np.concatenate([seed_indices, neighbor_indices])

                # Update remaining (exclude seed + neighbors)
                all_stratified_set = set(stratified_indices)
                remaining_indices = np.array([idx for idx in indices if idx not in all_stratified_set])

                logging.info(f"Enhanced stratified sample size: {len(stratified_indices)} "
                             f"(seed + neighbors)")
                logging.info(f"  - Seed nodes: {len(seed_indices)}")
                logging.info(f"  - Neighbor nodes added: {len(neighbor_indices)}")
                logging.info(f"  - Expansion ratio: {len(stratified_indices) / len(seed_indices):.2f}x")
            else:
                logging.warning("No neighbors found in dataset, using seed nodes only")
                stratified_indices = seed_indices
        else:
            if self.include_neighbors:
                logging.info("Neighborhood expansion enabled but no edges provided")
            else:
                logging.info("No neighborhood expansion (using seed nodes only)")
            stratified_indices = seed_indices

        # Step 3: Extract stratified and remaining data
        stratified_embeddings = embeddings[stratified_indices]
        stratified_ids = [node_ids[i] for i in stratified_indices]
        stratified_labels = class_labels[stratified_indices]

        remaining_embeddings = embeddings[remaining_indices]
        remaining_ids = [node_ids[i] for i in remaining_indices]
        remaining_labels = class_labels[remaining_indices]

        logging.info(f"Final stratified sample size: {len(stratified_ids)} "
                     f"({len(stratified_ids) / len(node_ids):.1%})")
        logging.info(f"Remaining sample size: {len(remaining_ids)} "
                     f"({len(remaining_ids) / len(node_ids):.1%})")

        # Verify seed class distribution (neighbors may have different distribution)
        seed_labels = class_labels[seed_indices]
        original_dist = dict(zip(*np.unique(class_labels, return_counts=True)))
        seed_dist = dict(zip(*np.unique(seed_labels, return_counts=True)))

        logging.info("Original class distribution (counts): %s", original_dist)
        logging.info("Seed class distribution (counts): %s", seed_dist)

        if self.include_neighbors and len(stratified_indices) > len(seed_indices):
            stratified_dist = dict(zip(*np.unique(stratified_labels, return_counts=True)))
            logging.info("Enhanced stratified class distribution (counts): %s", stratified_dist)

        log_memory_usage("After stratified sampling")

        return (stratified_embeddings, stratified_ids, stratified_labels,
                remaining_embeddings, remaining_ids, remaining_labels)


class CitationGraph:
    """Build and manage citation graph for spectral embeddings."""

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
        logging.info(f"Building induced citation subgraph for {len(node_ids)} nodes...")
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

        # Add self-loops to prevent singularity
        self.adjacency_matrix.setdiag(1)
        self.adjacency_matrix.eliminate_zeros()

        logging.info(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        logging.info(f"Number of edges (symmetric): {(self.adjacency_matrix.nnz - n_nodes) // 2}")
        logging.info(f"Graph density: {self.adjacency_matrix.nnz / (n_nodes * n_nodes):.6f}")

        # Check graph connectivity
        avg_degree = (self.adjacency_matrix.nnz - n_nodes) / n_nodes
        logging.info(f"Average node degree: {avg_degree:.2f}")

        if avg_degree < 2:
            logging.warning(f"WARNING: Graph is very sparse (avg degree: {avg_degree:.2f}). "
                            f"This may affect clustering quality.")

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

    def compute_citation_similarity_batched(self,
                                            remaining_nodes: List[int],
                                            stratified_nodes: List[int],
                                            batch_size: int = 1000) -> np.ndarray:
        """
        Compute cosine similarity between remaining and stratified nodes based on citations.
        Uses BATCHING to avoid memory issues.

        Args:
            remaining_nodes: List of remaining node IDs
            stratified_nodes: List of stratified node IDs
            batch_size: Number of remaining nodes to process at once

        Returns:
            Similarity matrix (n_remaining x n_stratified)
        """
        logging.info(f"Computing citation similarity for {len(remaining_nodes)} remaining nodes "
                     f"using batch_size={batch_size}...")
        log_memory_usage("Before citation similarity (batched)")

        n_remaining = len(remaining_nodes)
        n_stratified = len(stratified_nodes)

        # Pre-compute stratified vectors ONCE (much smaller - only 10%)
        logging.info(f"Pre-computing {n_stratified} stratified citation vectors...")
        stratified_vectors = np.array([self.get_citation_vector(node) for node in stratified_nodes])
        logging.info(f"Stratified vectors shape: {stratified_vectors.shape}")
        log_memory_usage("After stratified vectors")

        # Initialize result matrix
        similarity_matrix = np.zeros((n_remaining, n_stratified), dtype=np.float32)

        # Process remaining nodes in batches
        n_batches = (n_remaining + batch_size - 1) // batch_size
        logging.info(f"Processing {n_remaining} remaining nodes in {n_batches} batches...")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_remaining)
            batch_nodes = remaining_nodes[start_idx:end_idx]

            # Build citation vectors for this batch only
            batch_vectors = np.array([self.get_citation_vector(node) for node in batch_nodes])

            # Compute similarity for this batch
            batch_similarity = cosine_similarity(batch_vectors, stratified_vectors)

            # Store in result matrix
            similarity_matrix[start_idx:end_idx, :] = batch_similarity

            # Log progress
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                logging.info(f"Processed batch {batch_idx + 1}/{n_batches} "
                             f"({end_idx}/{n_remaining} nodes, "
                             f"{(end_idx / n_remaining) * 100:.1f}%)")
                log_memory_usage(f"After batch {batch_idx + 1}")

            # Clean up batch vectors
            del batch_vectors
            gc.collect()

        log_memory_usage("After citation similarity (batched)")
        logging.info("Citation similarity computation completed")

        return similarity_matrix


class SpectralEmbeddingGenerator:
    """Generate spectral embeddings from citation graph."""

    def __init__(self, n_spectral_dims: int = 50):
        """
        Initialize spectral embedding generator.

        Args:
            n_spectral_dims: Dimensionality of spectral embeddings
        """
        self.n_spectral_dims = n_spectral_dims
        self.spectral_embeddings = None

    def create_spectral_embeddings(self, adjacency_matrix: csr_matrix) -> np.ndarray:
        """
        Create spectral embeddings from adjacency matrix.

        Args:
            adjacency_matrix: Sparse adjacency matrix (symmetric, with self-loops)

        Returns:
            Spectral embeddings (n_nodes x n_spectral_dims)
        """
        logging.info(f"Creating spectral embeddings (dim={self.n_spectral_dims})...")
        log_memory_usage("Before spectral embedding")

        n_nodes = adjacency_matrix.shape[0]

        # Compute degree matrix
        degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()

        # Check for isolated nodes
        isolated_nodes = np.where(degrees == 0)[0]
        if len(isolated_nodes) > 0:
            logging.warning(f"Found {len(isolated_nodes)} isolated nodes")
            degrees[isolated_nodes] = 1

        # Compute D^(-1/2)
        D_inv_sqrt = np.sqrt(1.0 / degrees)
        D_inv_sqrt_diag = csr_matrix(np.diag(D_inv_sqrt))

        # Compute normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
        logging.info("Computing normalized Laplacian...")
        normalized_adjacency = D_inv_sqrt_diag @ adjacency_matrix @ D_inv_sqrt_diag

        # L_sym = I - normalized_adjacency
        identity = csr_matrix(np.eye(n_nodes))
        laplacian = identity - normalized_adjacency

        logging.info(f"Normalized Laplacian: shape={laplacian.shape}")

        # Compute eigenvectors
        logging.info(f"Computing {self.n_spectral_dims} smallest eigenvectors...")

        try:
            # Primary method: eigsh with normalized Laplacian
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=min(self.n_spectral_dims + 1, n_nodes - 2),
                which='SM',  # Smallest magnitude
                maxiter=5000,
                tol=1e-5,
                return_eigenvectors=True
            )

            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Remove first eigenvector (corresponding to eigenvalue ≈ 0)
            self.spectral_embeddings = eigenvectors[:, 1:min(self.n_spectral_dims + 1, eigenvectors.shape[1])]

            logging.info(f"Spectral embeddings created: shape={self.spectral_embeddings.shape}")
            logging.info(f"Eigenvalue range: [{eigenvalues[1]:.6f}, {eigenvalues[-1]:.6f}]")

        except Exception as e:
            logging.error(f"Primary eigenvalue computation failed: {e}")
            logging.warning("Trying alternative: Random-walk Laplacian")

            try:
                # Fallback 1: Random-walk Laplacian L_rw = I - D^(-1) * A
                D_inv = csr_matrix(np.diag(1.0 / degrees))
                laplacian_rw = identity - D_inv @ adjacency_matrix

                eigenvalues, eigenvectors = eigsh(
                    laplacian_rw,
                    k=min(self.n_spectral_dims + 1, n_nodes - 2),
                    which='SM',
                    maxiter=5000,
                    tol=1e-5,
                    return_eigenvectors=True
                )

                idx = np.argsort(eigenvalues)
                eigenvectors = eigenvectors[:, idx]
                self.spectral_embeddings = eigenvectors[:, 1:min(self.n_spectral_dims + 1, eigenvectors.shape[1])]

                logging.info(f"Random-walk Laplacian succeeded: shape={self.spectral_embeddings.shape}")

            except Exception as e2:
                logging.error(f"Random-walk Laplacian also failed: {e2}")
                logging.warning("Falling back to degree-based features")

                # Fallback 2: Use degree and adjacency statistics
                degree_features = degrees.reshape(-1, 1)
                adjacency_sum = np.array(adjacency_matrix.sum(axis=1))

                # Normalize features
                degree_features = normalize(degree_features, norm='l2')
                adjacency_sum = normalize(adjacency_sum, norm='l2')

                # Add random features to reach desired dimensionality
                random_features = np.random.randn(n_nodes, self.n_spectral_dims - 2)
                self.spectral_embeddings = np.hstack([degree_features, adjacency_sum, random_features])

                logging.info(f"Using fallback features: shape={self.spectral_embeddings.shape}")

        # Clean up
        del adjacency_matrix, normalized_adjacency, laplacian, D_inv_sqrt_diag
        gc.collect()

        log_memory_usage("After spectral embedding")

        return self.spectral_embeddings


class MultiviewMiniBatchSpectralClustering:
    """
    Multiview spectral clustering combining text embeddings and citation structure
    with mini-batch strategy for scalability.
    """

    def __init__(self,
                 embeddings_file: str,
                 nodes_csv: str,
                 edges_csv: str,
                 n_clusters: int = 40,
                 n_spectral_dims: int = 50,
                 graph_k: int = 20,
                 projection_k: int = 3,
                 alpha: float = 0.5,
                 stratified_ratio: float = 0.01,
                 include_neighbors: bool = True,
                 citation_batch_size: int = 1000,
                 output_dir: str = "results/multiview_minibatch_aligned",
                 random_state: int = 42):
        """
        Initialize multiview mini-batch spectral clustering pipeline.

        Args:
            embeddings_file: Path to embeddings .npy file with labels
            nodes_csv: Path to nodes CSV file (columns: node_id, class_idx)
            edges_csv: Path to edges CSV file (columns: source, target)
            n_clusters: Number of clusters
            n_spectral_dims: Dimensionality of citation spectral embeddings
            graph_k: Number of neighbors for k-NN graph in multiview clustering
            projection_k: Number of neighbors for projecting remaining nodes
            alpha: Weight for text view (0 to 1), (1-alpha) for spectral view
            stratified_ratio: Ratio of stratified sample (e.g., 0.01 for 1%)
            include_neighbors: Include 1-hop neighbors of stratified sample
            citation_batch_size: Batch size for computing citation similarity (to avoid OOM)
            output_dir: Output directory for results
            random_state: Random state for reproducibility
        """
        self.embeddings_file = embeddings_file
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.n_clusters = n_clusters
        self.n_spectral_dims = n_spectral_dims
        self.graph_k = graph_k
        self.projection_k = projection_k
        self.alpha = alpha
        self.stratified_ratio = stratified_ratio
        self.include_neighbors = include_neighbors
        self.citation_batch_size = citation_batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Data containers
        self.embeddings = None
        self.node_ids = None
        self.class_labels = None
        self.nodes_df = None
        self.edges_df = None

        # Stratified subsets
        self.stratified_embeddings = None
        self.stratified_ids = None
        self.stratified_labels = None
        self.remaining_embeddings = None
        self.remaining_ids = None
        self.remaining_labels = None

        # View matrices
        self.stratified_text_features = None
        self.stratified_spectral_features = None

        # Components
        self.sampler = StratifiedSampler(stratified_ratio, include_neighbors, random_state)
        self.citation_graph = None
        self.spectral_generator = SpectralEmbeddingGenerator(n_spectral_dims)
        self.citation_vectorizer = None

        # Clustering results
        self.cluster_labels_stratified = None
        self.text_centroids = None
        self.spectral_centroids = None
        self.final_cluster_labels = None

        # Setup logging
        self._setup_logging()

        logging.info(f"Initialized multiview mini-batch spectral clustering")
        logging.info(f"Parameters: n_clusters={n_clusters}, n_spectral_dims={n_spectral_dims}, "
                     f"graph_k={graph_k}, projection_k={projection_k}, alpha={alpha}, "
                     f"citation_batch_size={citation_batch_size}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"multiview_minibatch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_data(self):
        """Load embeddings and citation network data."""
        logging.info("Loading data...")
        log_memory_usage("Before loading data")

        # Load embeddings with labels
        data = np.load(self.embeddings_file)
        self.embeddings = data[:, :-2]
        self.node_ids = data[:, -2].astype(int).tolist()
        self.class_labels = data[:, -1].astype(int)
        del data
        gc.collect()

        logging.info(f"Loaded {len(self.embeddings)} embeddings (dim={self.embeddings.shape[1]})")
        logging.info(f"Number of unique classes: {len(np.unique(self.class_labels))}")

        # Load citation network
        self.nodes_df = pd.read_csv(self.nodes_csv)
        self.edges_df = pd.read_csv(self.edges_csv)

        logging.info(f"Loaded {len(self.nodes_df)} nodes and {len(self.edges_df)} edges")

        # Align datasets
        self._align_datasets()

        log_memory_usage("After loading data")

    def _align_datasets(self):
        """Ensure embeddings and citation network have same nodes in same order."""
        logging.info("Aligning datasets...")

        embedding_nodes = set(self.node_ids)
        citation_nodes = set(self.nodes_df['node_id'].tolist())
        common_nodes = embedding_nodes.intersection(citation_nodes)

        logging.info(f"Common nodes: {len(common_nodes)}")

        # Filter to common nodes
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        common_indices = [node_to_idx[node] for node in common_nodes if node in node_to_idx]

        self.embeddings = self.embeddings[common_indices]
        self.node_ids = [self.node_ids[i] for i in common_indices]
        self.class_labels = self.class_labels[common_indices]

        self.nodes_df = self.nodes_df[self.nodes_df['node_id'].isin(common_nodes)]

        # Sort nodes_df to match node_ids order
        node_id_to_position = {nid: pos for pos, nid in enumerate(self.node_ids)}
        self.nodes_df['_sort_order'] = self.nodes_df['node_id'].map(node_id_to_position)
        self.nodes_df = self.nodes_df.sort_values('_sort_order').drop('_sort_order', axis=1).reset_index(drop=True)

        logging.info(f"Aligned to {len(self.node_ids)} common nodes")

    def create_stratified_sample(self):
        """Create stratified sample with optional neighborhood expansion."""
        (self.stratified_embeddings, self.stratified_ids, self.stratified_labels,
         self.remaining_embeddings, self.remaining_ids, self.remaining_labels) = \
            self.sampler.create_stratified_sample(
                self.embeddings, self.node_ids, self.class_labels, self.edges_df
            )

    def prepare_views_on_stratified(self):
        """Prepare both views (text + spectral) on stratified subset."""
        logging.info("Preparing views on stratified subset...")
        log_memory_usage("Before view preparation")

        # View 1: Text embeddings (normalize)
        self.stratified_text_features = normalize(self.stratified_embeddings, norm='l2')
        logging.info(f"View 1 (Text): shape={self.stratified_text_features.shape}")

        # View 2: Spectral embeddings from citation subgraph
        self.citation_graph = CitationGraph(self.edges_df)
        adjacency_matrix = self.citation_graph.build_induced_subgraph(self.stratified_ids)

        # Check connectivity (same as citation-only - warn only, don't fix)
        n_nodes = len(self.stratified_ids)
        avg_degree = (adjacency_matrix.nnz - n_nodes) / n_nodes if n_nodes > 0 else 0
        logging.info(f"Average node degree: {avg_degree:.2f}")

        if avg_degree < 2:
            logging.warning(f"WARNING: Graph is very sparse (avg degree: {avg_degree:.2f}). "
                            f"This may affect clustering quality and convergence.")
            logging.warning("Consider: (1) Using larger stratified_ratio, or (2) Different clustering approach")

        self.stratified_spectral_features = self.spectral_generator.create_spectral_embeddings(adjacency_matrix)
        self.stratified_spectral_features = normalize(self.stratified_spectral_features, norm='l2')
        logging.info(f"View 2 (Citation Spectral): shape={self.stratified_spectral_features.shape}")

        # Verify alignment
        assert self.stratified_text_features.shape[0] == self.stratified_spectral_features.shape[0], \
            "Views must have same number of samples"

        log_memory_usage("After view preparation")

    def perform_multiview_clustering_on_stratified(self):
        """Perform multiview spectral clustering on stratified subset."""
        logging.info("Performing multiview spectral clustering on stratified subset...")
        log_memory_usage("Before multiview clustering")

        views = [self.stratified_text_features, self.stratified_spectral_features]

        try:
            if MVLEARN_AVAILABLE:
                # Use mvlearn's MultiviewSpectralClustering
                logging.info("Using mvlearn's MultiviewSpectralClustering")
                mvsc = MultiviewSpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=self.graph_k,
                    random_state=self.random_state,
                    n_init=10
                )
                self.cluster_labels_stratified = mvsc.fit_predict(views)

            else:
                # Fallback: Concatenate views and use standard spectral clustering
                logging.warning("mvlearn not available. Using concatenated views.")
                from sklearn.cluster import SpectralClustering

                combined_features = np.hstack(views)
                sc = SpectralClustering(
                    n_clusters=self.n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=self.graph_k,
                    random_state=self.random_state,
                    n_init=10
                )
                self.cluster_labels_stratified = sc.fit_predict(combined_features)

            if self.cluster_labels_stratified is None or len(self.cluster_labels_stratified) == 0:
                raise RuntimeError("Clustering failed to produce labels")

            logging.info(f"Clustering complete: {len(np.unique(self.cluster_labels_stratified))} clusters found")

        except Exception as e:
            logging.error(f"Multiview clustering failed: {e}")
            raise

        log_memory_usage("After multiview clustering")
        gc.collect()

    def compute_multiview_centroids(self):
        """Compute cluster centroids in both text and spectral spaces."""
        logging.info("Computing multiview cluster centroids...")

        self.text_centroids = np.zeros((self.n_clusters, self.stratified_text_features.shape[1]))
        self.spectral_centroids = np.zeros((self.n_clusters, self.stratified_spectral_features.shape[1]))

        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_stratified == cluster_id

            # Text centroid
            cluster_text_points = self.stratified_text_features[cluster_mask]
            if len(cluster_text_points) > 0:
                self.text_centroids[cluster_id] = cluster_text_points.mean(axis=0)

            # Spectral centroid
            cluster_spectral_points = self.stratified_spectral_features[cluster_mask]
            if len(cluster_spectral_points) > 0:
                self.spectral_centroids[cluster_id] = cluster_spectral_points.mean(axis=0)

        logging.info(f"Computed {self.n_clusters} centroids in both spaces")
        logging.info(f"Text centroid shape: {self.text_centroids.shape}")
        logging.info(f"Spectral centroid shape: {self.spectral_centroids.shape}")

    def project_remaining_to_spectral_space(self, batch_size: int = 1000) -> np.ndarray:
        """
        Project remaining nodes to spectral space using citation similarity + k-NN.
        Uses BATCHING to avoid memory issues.

        Args:
            batch_size: Number of nodes to process at once for citation similarity

        Returns:
            Spectral embeddings for remaining nodes
        """
        logging.info(f"Projecting {len(self.remaining_ids)} remaining nodes to spectral space...")
        log_memory_usage("Before spectral projection")

        # Build citation vectorizer
        all_node_ids = self.node_ids
        self.citation_vectorizer = CitationVectorizer(self.edges_df, all_node_ids)

        # Compute citation similarity between remaining and stratified nodes (BATCHED!)
        similarity_matrix = self.citation_vectorizer.compute_citation_similarity_batched(
            self.remaining_ids, self.stratified_ids, batch_size=batch_size
        )

        # Project using k-NN weighted averaging
        n_remaining = len(self.remaining_ids)
        remaining_spectral = np.zeros((n_remaining, self.stratified_spectral_features.shape[1]))

        logging.info(f"Projecting {n_remaining} nodes to spectral space using k={self.projection_k}...")

        for i in range(n_remaining):
            similarities = similarity_matrix[i]

            # Find k nearest neighbors
            if np.sum(similarities > 0) >= self.projection_k:
                top_k_indices = np.argpartition(similarities, -self.projection_k)[-self.projection_k:]
            else:
                positive_sim_indices = np.where(similarities > 0)[0]
                if len(positive_sim_indices) > 0:
                    top_k_indices = positive_sim_indices
                else:
                    top_k_indices = np.argpartition(similarities, -self.projection_k)[-self.projection_k:]

            # Get weights
            weights = similarities[top_k_indices]

            # Normalize weights
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(top_k_indices)) / len(top_k_indices)

            # Weighted average
            for j, neighbor_idx in enumerate(top_k_indices):
                remaining_spectral[i] += weights[j] * self.stratified_spectral_features[neighbor_idx]

            # Log progress
            if (i + 1) % 10000 == 0 or i == n_remaining - 1:
                logging.info(f"Projected {i + 1}/{n_remaining} nodes ({(i + 1) / n_remaining * 100:.1f}%)")

        log_memory_usage("After spectral projection")
        logging.info("Spectral projection completed")

        return remaining_spectral

    def assign_remaining_with_combined_similarity(self, remaining_spectral: np.ndarray) -> np.ndarray:
        """
        Assign remaining nodes to clusters using combined text + spectral similarity.

        Args:
            remaining_spectral: Spectral embeddings for remaining nodes

        Returns:
            Cluster labels for remaining nodes
        """
        logging.info(f"Assigning remaining nodes using combined similarity (alpha={self.alpha})...")
        log_memory_usage("Before assignment")

        # Normalize remaining text embeddings
        remaining_text = normalize(self.remaining_embeddings, norm='l2')
        remaining_spectral = normalize(remaining_spectral, norm='l2')

        n_remaining = len(remaining_text)
        cluster_labels_remaining = np.zeros(n_remaining, dtype=int)

        for i in range(n_remaining):
            best_cluster = -1
            best_similarity = -np.inf

            # Compute similarity to each cluster
            for cluster_id in range(self.n_clusters):
                # Text similarity
                text_sim = cosine_similarity(
                    remaining_text[i].reshape(1, -1),
                    self.text_centroids[cluster_id].reshape(1, -1)
                )[0, 0]

                # Spectral similarity
                spectral_sim = cosine_similarity(
                    remaining_spectral[i].reshape(1, -1),
                    self.spectral_centroids[cluster_id].reshape(1, -1)
                )[0, 0]

                # Combined similarity using alpha
                combined_sim = self.alpha * text_sim + (1 - self.alpha) * spectral_sim

                if combined_sim > best_similarity:
                    best_similarity = combined_sim
                    best_cluster = cluster_id

            cluster_labels_remaining[i] = best_cluster

        log_memory_usage("After assignment")
        logging.info(f"Assigned {len(cluster_labels_remaining)} nodes to clusters")

        return cluster_labels_remaining

    def run_clustering(self) -> Dict[str, Any]:
        """Run complete multiview mini-batch clustering pipeline."""
        logging.info("Starting multiview mini-batch spectral clustering pipeline...")
        log_memory_usage("Pipeline start")

        # Step 1: Load data
        self.load_data()

        # Step 2: Create stratified sample
        self.create_stratified_sample()

        # Step 3: Prepare views on stratified subset
        self.prepare_views_on_stratified()

        # Step 4: Perform multiview clustering on stratified subset
        self.perform_multiview_clustering_on_stratified()

        # Step 5: Compute multiview centroids
        self.compute_multiview_centroids()

        # Step 6: Project remaining nodes to spectral space
        remaining_spectral = self.project_remaining_to_spectral_space(
            batch_size=self.citation_batch_size
        )

        # Step 7: Assign remaining nodes using combined similarity
        remaining_cluster_labels = self.assign_remaining_with_combined_similarity(remaining_spectral)

        # Step 8: Reconstruct full labels
        self.final_cluster_labels = self._reconstruct_full_labels(
            self.cluster_labels_stratified, remaining_cluster_labels
        )

        # Step 9: Evaluate
        metrics = self.evaluate_clustering()

        # Step 10: Analyze composition
        composition_analysis = self.analyze_cluster_composition()

        # Step 11: Visualize
        self.visualize_results()

        # Step 12: Prepare results
        results = {
            'method': 'multiview_minibatch_spectral_clustering',
            'embeddings_file': str(self.embeddings_file),
            'nodes_csv': str(self.nodes_csv),
            'edges_csv': str(self.edges_csv),
            'stratified_ratio': self.stratified_ratio,
            'n_spectral_dims': self.n_spectral_dims,
            'graph_k': self.graph_k,
            'projection_k': self.projection_k,
            'alpha': self.alpha,
            'n_clusters': self.n_clusters,
            'n_papers_total': len(self.node_ids),
            'n_papers_stratified': len(self.stratified_ids),
            'n_papers_remaining': len(self.remaining_ids),
            'embedding_dim': self.embeddings.shape[1],
            'n_edges_total': len(self.edges_df),
            'evaluation_metrics': metrics,
            'composition_analysis': composition_analysis,
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        results_file = self.output_dir / "clustering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_file}")

        # Print formatted results
        self.print_results_formatted(results)

        log_memory_usage("Pipeline end")
        logging.info("Multiview mini-batch clustering pipeline completed successfully!")

        return results

    def _reconstruct_full_labels(self,
                                 stratified_labels: np.ndarray,
                                 remaining_labels: np.ndarray) -> np.ndarray:
        """Reconstruct full label array in original order."""
        logging.info("Reconstructing full label array...")

        label_dict = {}

        for node_id, label in zip(self.stratified_ids, stratified_labels):
            label_dict[node_id] = label

        for node_id, label in zip(self.remaining_ids, remaining_labels):
            label_dict[node_id] = label

        full_labels = np.array([label_dict[node_id] for node_id in self.node_ids])

        logging.info(f"Reconstructed full labels array with {len(full_labels)} labels")

        return full_labels

    def evaluate_clustering(self) -> Dict[str, float]:
        """Evaluate clustering using ARI, NMI, and V-measure."""
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
        """Analyze cluster composition."""
        logging.info("Analyzing cluster composition...")

        cluster_composition = {}
        unique_clusters = np.unique(self.final_cluster_labels)
        unique_classes = np.unique(self.class_labels)

        for cluster_id in unique_clusters:
            cluster_mask = self.final_cluster_labels == cluster_id
            cluster_ground_truth = self.class_labels[cluster_mask]

            class_counts = {}
            for class_label in unique_classes:
                count = np.sum(cluster_ground_truth == class_label)
                if count > 0:
                    class_counts[int(class_label)] = int(count)

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

        self._print_cluster_composition_analysis(cluster_composition, summary_stats)
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

            sorted_classes = sorted(comp['class_counts'].items(), key=lambda x: x[1], reverse=True)
            print("  Class distribution:")
            for class_id, count in sorted_classes[:3]:
                percentage = comp['class_percentages'][class_id]
                print(f"    Class {class_id}: {count} papers ({percentage:.1f}%)")

            if len(sorted_classes) > 3:
                print(f"    ... and {len(sorted_classes) - 3} more classes")

    def _save_cluster_composition_to_csv(self, cluster_composition: Dict):
        """Save detailed cluster composition to CSV."""
        try:
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

            df = pd.DataFrame(csv_data)
            csv_path = self.output_dir / "cluster_composition_detailed.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"Detailed cluster composition saved to {csv_path}")
            print(f"\n✓ Detailed composition saved to {csv_path}")

        except Exception as e:
            logging.warning(f"Could not save cluster composition to CSV: {e}")

    def visualize_results(self):
        """Create visualizations."""
        logging.info("Creating visualizations...")

        self._plot_distribution_comparison()
        self._plot_confusion_matrix()
        self._plot_cluster_sizes()

        logging.info("Visualizations completed")

    def _plot_distribution_comparison(self):
        """Plot ground truth vs predicted distribution."""
        log_memory_usage("Before distribution plot")

        gt_unique, gt_counts = np.unique(self.class_labels, return_counts=True)
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

        plt.suptitle('Ground Truth vs Predicted Distribution (Multiview MiniBatch)',
                     fontsize=14, fontweight='bold')
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

        plt.title('Confusion Matrix: Ground Truth vs Predicted Clusters (Multiview MiniBatch)')
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
        plt.title('Cluster Size Distribution (Multiview MiniBatch)')
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
        print("MULTIVIEW MINI-BATCH SPECTRAL CLUSTERING RESULTS")
        print("=" * 80)
        print(f"Method: {results['method']}")
        print(f"Stratified ratio: {results['stratified_ratio']:.1%}")
        print(f"Spectral embedding dims: {results['n_spectral_dims']}")
        print(f"Graph construction k: {results['graph_k']}")
        print(f"Projection k: {results['projection_k']}")
        print(f"Alpha (text weight): {results['alpha']:.2f}")
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
    """Main function with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multiview Mini-batch Spectral Clustering for Citation Networks"
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        required=True,
        help="Path to embeddings .npy file with labels [embedding_dims, node_id, class_idx]"
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
        "--n-spectral-dims",
        type=int,
        default=50,
        help="Dimensionality of citation spectral embeddings (default: 50)"
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=20,
        help="Number of neighbors for k-NN graph in multiview clustering (default: 20)"
    )
    parser.add_argument(
        "--projection-k",
        type=int,
        default=3,
        help="Number of neighbors for projecting remaining nodes (default: 3)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for text view, (1-alpha) for spectral view (default: 0.5)"
    )
    parser.add_argument(
        "--stratified-ratio",
        type=float,
        default=0.01,
        help="Ratio of stratified sample (default: 0.01 for 1%%)"
    )
    parser.add_argument(
        "--include-neighbors",
        action="store_true",
        default=True,
        help="Include 1-hop neighbors of stratified sample (default: True)"
    )
    parser.add_argument(
        "--no-neighbors",
        action="store_false",
        dest="include_neighbors",
        help="Disable neighborhood expansion"
    )
    parser.add_argument(
        "--citation-batch-size",
        type=int,
        default=1000,
        help="Batch size for computing citation similarity to avoid OOM (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multiview_minibatch_aligned",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Validate alpha
    if not 0 <= args.alpha <= 1:
        raise ValueError("Alpha must be between 0 and 1")

    # Create and run pipeline
    pipeline = MultiviewMiniBatchSpectralClustering(
        embeddings_file=args.embeddings_file,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        n_clusters=args.n_clusters,
        n_spectral_dims=args.n_spectral_dims,
        graph_k=args.graph_k,
        projection_k=args.projection_k,
        alpha=args.alpha,
        stratified_ratio=args.stratified_ratio,
        include_neighbors=args.include_neighbors,
        citation_batch_size=args.citation_batch_size,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    results = pipeline.run_clustering()

    # Print summary
    print("\n" + "=" * 80)
    print("MULTIVIEW MINI-BATCH SPECTRAL CLUSTERING COMPLETED")
    print("=" * 80)
    print(f"ARI: {results['evaluation_metrics']['adjusted_rand_index']:.4f}")
    print(f"NMI: {results['evaluation_metrics']['normalized_mutual_info']:.4f}")
    print(f"V-measure: {results['evaluation_metrics']['v_measure']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()