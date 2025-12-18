import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import seaborn as sns
from mvlearn.cluster import MultiviewSpectralClustering
from pathlib import Path
import logging
import json
from datetime import datetime
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiviewCitationClustering:
    """
    Multiview spectral clustering combining text embeddings and citation structure.
    Uses mvlearn's MultiviewSpectralClustering with proper feature matrices.
    """

    def __init__(
            self,
            embeddings_file: str,
            nodes_csv: str,
            edges_csv: str,
            n_clusters: int = 40,
            n_spectral_dims: int = 50,  # NEW: dimensionality for spectral embedding
            k_neighbors: int = 3,
            output_dir: str = "results/multiview_clustering",
            random_state: int = 42
    ):
        self.embeddings_file = embeddings_file
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.n_clusters = n_clusters
        self.n_spectral_dims = n_spectral_dims  # NEW
        self.k_neighbors = k_neighbors
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Data containers
        self.embeddings = None
        self.node_ids = None
        self.class_labels = None
        self.nodes_df = None
        self.edges_df = None

        # Feature matrices for multiview
        self.text_features = None  # View 1: Text embeddings
        self.citation_features = None  # View 2: Spectral embeddings from citation graph

        # Results
        self.cluster_labels = None
        self.results = {}

        logging.info(
            f"Initialized with n_clusters={n_clusters}, n_spectral_dims={n_spectral_dims}, k_neighbors={k_neighbors}")

    def load_data(self):
        """Load embeddings and citation network data."""
        logging.info("Loading data...")

        # Load embeddings with labels
        data = np.load(self.embeddings_file)
        self.embeddings = data[:, :-2]
        self.node_ids = data[:, -2].astype(int).tolist()
        self.class_labels = data[:, -1].astype(int)
        del data
        gc.collect()

        logging.info(f"Loaded {len(self.embeddings)} embeddings (dim={self.embeddings.shape[1]})")
        logging.info(f"Ground truth classes: {len(np.unique(self.class_labels))}")

        # Load citation network
        self.nodes_df = pd.read_csv(self.nodes_csv)
        self.edges_df = pd.read_csv(self.edges_csv)

        logging.info(f"Loaded {len(self.nodes_df)} nodes and {len(self.edges_df)} citation edges")

        # Align datasets
        self._align_datasets()

    def _align_datasets(self):
        """Ensure embeddings and citation network have same nodes in same order."""
        logging.info("Aligning datasets...")

        embedding_nodes = set(self.node_ids)
        citation_nodes = set(self.nodes_df['node_id'].tolist())
        common_nodes = embedding_nodes.intersection(citation_nodes)

        logging.info(f"Common nodes: {len(common_nodes)}")

        # Filter to common nodes and maintain consistent ordering
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        common_indices = [node_to_idx[node] for node in common_nodes if node in node_to_idx]

        self.embeddings = self.embeddings[common_indices]
        self.node_ids = [self.node_ids[i] for i in common_indices]
        self.class_labels = self.class_labels[common_indices]

        self.nodes_df = self.nodes_df[self.nodes_df['node_id'].isin(common_nodes)]

        # Sort nodes_df to match node_ids order for consistency
        node_id_to_position = {nid: pos for pos, nid in enumerate(self.node_ids)}
        self.nodes_df['_sort_order'] = self.nodes_df['node_id'].map(node_id_to_position)
        self.nodes_df = self.nodes_df.sort_values('_sort_order').drop('_sort_order', axis=1).reset_index(drop=True)

        logging.info(f"Aligned to {len(self.node_ids)} common nodes")

    def create_spectral_embedding_from_citations(self):
        """
        Create spectral embedding features from citation graph.
        Handles isolated nodes by adding self-loops or using modified Laplacian.
        """
        logging.info(f"Creating spectral embedding from citation graph (dim={self.n_spectral_dims})...")

        # Build adjacency matrix
        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        n_nodes = len(self.node_ids)

        valid_edges = self.edges_df[
            (self.edges_df['source'].isin(node_to_idx.keys())) &
            (self.edges_df['target'].isin(node_to_idx.keys()))
            ]

        logging.info(f"Valid citation edges: {len(valid_edges)}")

        # Create symmetric adjacency matrix (undirected)
        row_indices = []
        col_indices = []

        for _, edge in valid_edges.iterrows():
            src_idx = node_to_idx[edge['source']]
            tgt_idx = node_to_idx[edge['target']]

            row_indices.extend([src_idx, tgt_idx])
            col_indices.extend([tgt_idx, src_idx])

        data = np.ones(len(row_indices))
        adjacency = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes)
        )

        # ADD SELF-LOOPS TO ALL NODES (prevents singularity)
        adjacency.setdiag(1)
        adjacency.eliminate_zeros()

        logging.info(f"Adjacency matrix: shape={adjacency.shape}, nnz={adjacency.nnz}")

        # Compute degree matrix
        degrees = np.array(adjacency.sum(axis=1)).flatten()

        # Check for isolated nodes (should be none now with self-loops)
        isolated_nodes = np.where(degrees == 0)[0]
        if len(isolated_nodes) > 0:
            logging.warning(f"Found {len(isolated_nodes)} isolated nodes (degree=0)")
            degrees[isolated_nodes] = 1

        # Compute D^(-1/2)
        D_inv_sqrt = np.sqrt(1.0 / degrees)
        D_inv_sqrt_diag = csr_matrix(np.diag(D_inv_sqrt))

        # Compute normalized Laplacian: L_sym = I - D^(-1/2) * A * D^(-1/2)
        logging.info("Computing normalized Laplacian...")
        normalized_adjacency = D_inv_sqrt_diag @ adjacency @ D_inv_sqrt_diag

        # L_sym = I - normalized_adjacency
        identity = csr_matrix(np.eye(n_nodes))
        laplacian = identity - normalized_adjacency

        logging.info(f"Normalized Laplacian: shape={laplacian.shape}")

        # Compute eigenvectors using different method for robustness
        logging.info(f"Computing {self.n_spectral_dims} smallest eigenvectors...")

        try:
            # Use 'LM' (largest magnitude) on -Laplacian to get smallest eigenvalues
            # More stable than 'SM' with sigma
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=min(self.n_spectral_dims + 1, n_nodes - 2),  # Ensure k < n-1
                which='SM',  # Smallest magnitude
                maxiter=5000,  # Increase iterations
                tol=1e-5,  # Relaxed tolerance
                return_eigenvectors=True
            )

            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Remove the first eigenvector (corresponding to eigenvalue ≈ 0)
            self.citation_features = eigenvectors[:, 1:min(self.n_spectral_dims + 1, eigenvectors.shape[1])]

            logging.info(f"Spectral embedding created: shape={self.citation_features.shape}")
            logging.info(f"Eigenvalue range: [{eigenvalues[1]:.6f}, {eigenvalues[-1]:.6f}]")

        except Exception as e:
            logging.error(f"Eigenvalue computation failed: {e}")
            logging.warning("Trying alternative: Random-walk Laplacian")

            try:
                # Alternative: Use random-walk Laplacian L_rw = I - D^(-1) * A
                D_inv = csr_matrix(np.diag(1.0 / degrees))
                laplacian_rw = identity - D_inv @ adjacency

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
                self.citation_features = eigenvectors[:, 1:min(self.n_spectral_dims + 1, eigenvectors.shape[1])]

                logging.info(f"Random-walk Laplacian succeeded: shape={self.citation_features.shape}")

            except Exception as e2:
                logging.error(f"Random-walk Laplacian also failed: {e2}")
                logging.warning("Falling back to adjacency-based features")

                # Final fallback: Use degree and adjacency statistics
                degree_features = degrees.reshape(-1, 1)
                adjacency_sum = np.array(adjacency.sum(axis=1))

                # Add some basic graph features
                self.citation_features = np.hstack([
                    degree_features,
                    adjacency_sum,
                    np.random.randn(n_nodes, self.n_spectral_dims - 2)
                ])

                logging.info(f"Using fallback features: shape={self.citation_features.shape}")

        del adjacency, normalized_adjacency, laplacian, D_inv_sqrt_diag
        gc.collect()

    def prepare_views(self):
        """Prepare feature matrices for both views."""
        logging.info("Preparing views for multiview clustering...")

        # View 1: Text embeddings (already available)
        self.text_features = self.embeddings

        # Normalize text features (L2 normalization)
        from sklearn.preprocessing import normalize
        self.text_features = normalize(self.text_features, norm='l2')

        # View 2: Spectral embeddings from citation graph
        self.create_spectral_embedding_from_citations()

        # Normalize citation features
        self.citation_features = normalize(self.citation_features, norm='l2')

        logging.info(f"View 1 (Text): shape={self.text_features.shape}")
        logging.info(f"View 2 (Citation Spectral): shape={self.citation_features.shape}")

        # Verify alignment
        assert self.text_features.shape[0] == self.citation_features.shape[0], \
            "Views must have same number of samples"
        assert len(self.node_ids) == self.text_features.shape[0], \
            "Node IDs must match number of samples"

    def perform_multiview_clustering(self):
        """Perform multiview spectral clustering using mvlearn."""
        logging.info("Performing multiview spectral clustering with mvlearn...")

        # Create list of views
        views = [self.text_features, self.citation_features]

        logging.info(f"View shapes: {[v.shape for v in views]}")

        try:
            # Initialize multiview spectral clustering
            mvsc = MultiviewSpectralClustering(
                n_clusters=self.n_clusters,
                affinity='nearest_neighbors',  # Can also try 'rbf'
                n_neighbors=self.k_neighbors,
                random_state=self.random_state,
                n_init=10
            )

            # Fit and predict
            self.cluster_labels = mvsc.fit_predict(views)

            if self.cluster_labels is None or len(self.cluster_labels) == 0:
                raise RuntimeError("Clustering failed to produce labels")

            logging.info(f"Clustering complete: {len(np.unique(self.cluster_labels))} clusters found")

        except Exception as e:
            logging.error(f"mvlearn clustering failed: {e}")
            logging.warning("Falling back to standard spectral clustering on concatenated features")

            # Fallback: concatenate views and use standard spectral clustering
            from sklearn.cluster import SpectralClustering
            combined_features = np.hstack([self.text_features, self.citation_features])

            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity='nearest_neighbors',
                n_neighbors=self.k_neighbors,
                random_state=self.random_state,
                n_init=10
            )

            self.cluster_labels = sc.fit_predict(combined_features)

        gc.collect()

    def evaluate_clustering(self):
        """Evaluate clustering against ground truth."""
        logging.info("Evaluating clustering...")

        if self.cluster_labels is None:
            raise RuntimeError("No cluster labels found")

        ari = adjusted_rand_score(self.class_labels, self.cluster_labels)
        nmi = normalized_mutual_info_score(self.class_labels, self.cluster_labels)

        # Calculate purity
        purities = []
        for cluster_id in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cluster_id
            cluster_gt = self.class_labels[mask]
            most_common = np.bincount(cluster_gt.astype(int)).argmax()
            purity = np.mean(cluster_gt == most_common)
            purities.append(purity)

        mean_purity = np.mean(purities)

        # Detailed composition analysis
        composition = self._analyze_cluster_composition()

        self.results = {
            'adjusted_rand_index': float(ari),
            'normalized_mutual_info': float(nmi),
            'mean_cluster_purity': float(mean_purity),
            'n_clusters': int(self.n_clusters),
            'n_spectral_dims': int(self.n_spectral_dims),
            'n_papers': int(len(self.cluster_labels)),
            'n_ground_truth_classes': int(len(np.unique(self.class_labels))),
            'composition_analysis': composition,
            'timestamp': datetime.now().isoformat()
        }

        logging.info(f"ARI: {ari:.4f}")
        logging.info(f"NMI: {nmi:.4f}")
        logging.info(f"Mean Purity: {mean_purity:.4f}")

        return self.results

    def _analyze_cluster_composition(self):
        """Analyze detailed cluster composition."""
        logging.info("Analyzing cluster composition...")

        print("\n" + "=" * 80)
        print("DETAILED CLUSTER COMPOSITION ANALYSIS")
        print("=" * 80)

        cluster_composition = {}
        unique_clusters = np.unique(self.cluster_labels)
        unique_classes = np.unique(self.class_labels)

        for cluster_id in unique_clusters:
            mask = self.cluster_labels == cluster_id
            cluster_gt = self.class_labels[mask]

            class_counts = {}
            for class_label in unique_classes:
                count = np.sum(cluster_gt == class_label)
                if count > 0:
                    class_counts[int(class_label)] = int(count)

            total = len(cluster_gt)
            class_percentages = {cid: (cnt / total) * 100 for cid, cnt in class_counts.items()}

            cluster_composition[int(cluster_id)] = {
                'total_papers': total,
                'class_counts': class_counts,
                'class_percentages': class_percentages,
                'num_different_classes': len(class_counts),
                'dominant_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None,
                'dominant_class_percentage': max(class_percentages.values()) if class_percentages else 0
            }

        summary_stats = {
            'total_clusters': len(unique_clusters),
            'total_classes': len(unique_classes),
            'avg_classes_per_cluster': np.mean([c['num_different_classes'] for c in cluster_composition.values()]),
            'avg_dominant_class_percentage': np.mean(
                [c['dominant_class_percentage'] for c in cluster_composition.values()]),
            'clusters_with_single_class': sum(
                1 for c in cluster_composition.values() if c['num_different_classes'] == 1),
            'clusters_with_multiple_classes': sum(
                1 for c in cluster_composition.values() if c['num_different_classes'] > 1)
        }

        # Print summary
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
            for class_id, count in sorted_classes[:5]:
                percentage = comp['class_percentages'][class_id]
                print(f"    Class {class_id}: {count} papers ({percentage:.1f}%)")

            if len(sorted_classes) > 5:
                print(f"    ... and {len(sorted_classes) - 5} more classes")

        # Save to CSV
        self._save_composition_csv(cluster_composition)

        return {
            'cluster_composition': cluster_composition,
            'summary_statistics': summary_stats
        }

    def _save_composition_csv(self, cluster_composition):
        """Save detailed composition to CSV."""
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
            csv_path = self.output_dir / "multiview_cluster_composition_detailed.csv"
            df.to_csv(csv_path, index=False)

            logging.info(f"Detailed composition saved to {csv_path}")
            print(f"\n✓ Detailed composition saved to {csv_path}")

        except Exception as e:
            logging.warning(f"Could not save CSV: {e}")

    def visualize_results(self):
        """Create visualizations of clustering results."""
        logging.info("Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Ground truth distribution
        axes[0, 0].hist(self.class_labels, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Ground Truth Class Distribution')
        axes[0, 0].set_xlabel('Class ID')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Predicted cluster distribution
        axes[0, 1].hist(self.cluster_labels, bins=30, alpha=0.7, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_title('Predicted Cluster Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Confusion matrix (subset)
        confusion = pd.crosstab(self.class_labels, self.cluster_labels)
        subset = confusion.iloc[:20, :20]
        sns.heatmap(subset, annot=False, cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix (20x20 Subset)')
        axes[1, 0].set_xlabel('Predicted Cluster')
        axes[1, 0].set_ylabel('True Class')

        # Evaluation metrics
        metrics_text = (
            f"Multiview Spectral Clustering (mvlearn)\n\n"
            f"Adjusted Rand Index: {self.results['adjusted_rand_index']:.4f}\n"
            f"Normalized Mutual Info: {self.results['normalized_mutual_info']:.4f}\n"
            f"Mean Cluster Purity: {self.results['mean_cluster_purity']:.4f}\n\n"
            f"Number of Clusters: {self.results['n_clusters']}\n"
            f"Spectral Embedding Dim: {self.results['n_spectral_dims']}\n"
            f"Number of Papers: {self.results['n_papers']}\n"
            f"Ground Truth Classes: {self.results['n_ground_truth_classes']}"
        )
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11,
                        verticalalignment='center', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 1].set_title('Evaluation Metrics')
        axes[1, 1].axis('off')

        plt.tight_layout()

        save_path = self.output_dir / "multiview_clustering_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Visualization saved to {save_path}")

    def save_results(self):
        """Save all results to files."""
        logging.info("Saving results...")

        # Save cluster assignments
        results_df = pd.DataFrame({
            'node_id': self.node_ids,
            'cluster_id': self.cluster_labels,
            'ground_truth_class': self.class_labels
        })
        results_df.to_csv(self.output_dir / "multiview_cluster_assignments.csv", index=False)

        # Save metrics JSON
        with open(self.output_dir / "multiview_metrics.json", 'w') as f:
            json.dump(self.results, f, indent=2)

        logging.info(f"All results saved to {self.output_dir}")

    def run_complete_pipeline(self):
        """Run the complete multiview clustering pipeline."""
        try:
            print("\n" + "=" * 60)
            print("MULTIVIEW SPECTRAL CLUSTERING PIPELINE (mvlearn)")
            print("=" * 60)

            self.load_data()
            self.prepare_views()
            self.perform_multiview_clustering()
            self.evaluate_clustering()
            self.visualize_results()
            self.save_results()

            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"Results saved to: {self.output_dir}")
            print("=" * 60)

            return self.results

        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Multiview Spectral Clustering for Citation Networks (mvlearn)")
    parser.add_argument("--embeddings-file", type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument("--nodes-csv", type=str, required=True, help="Path to nodes CSV file")
    parser.add_argument("--edges-csv", type=str, required=True, help="Path to edges CSV file")
    parser.add_argument("--n-clusters", type=int, default=40, help="Number of clusters")
    parser.add_argument("--n-spectral-dims", type=int, default=50, help="Spectral embedding dimensionality")
    parser.add_argument("--k-neighbors", type=int, default=20, help="Number of neighbors for k-NN graph")
    parser.add_argument("--output-dir", type=str, default="results/multiview_clustering_mvlearn",
                        help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    clustering = MultiviewCitationClustering(
        embeddings_file=args.embeddings_file,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        n_clusters=args.n_clusters,
        n_spectral_dims=args.n_spectral_dims,
        k_neighbors=args.k_neighbors,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    results = clustering.run_complete_pipeline()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Adjusted Rand Index: {results['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Info: {results['normalized_mutual_info']:.4f}")
    print(f"Mean Cluster Purity: {results['mean_cluster_purity']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()