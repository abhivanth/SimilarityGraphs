import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from pathlib import Path
import logging
import json
from datetime import datetime
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiviewCitationClustering:
    """
    Multiview spectral clustering combining text embeddings and citation structure.
    """

    def __init__(
            self,
            embeddings_file: str,
            nodes_csv: str,
            edges_csv: str,
            n_clusters: int = 40,
            k_neighbors: int = 20,
            output_dir: str = "results/multiview_clustering",
            random_state: int = 42
    ):
        self.embeddings_file = embeddings_file
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.n_clusters = n_clusters
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

        # Affinity matrices
        self.text_affinity = None
        self.citation_affinity = None

        # Results
        self.cluster_labels = None
        self.results = {}

        logging.info(f"Initialized with n_clusters={n_clusters}, k_neighbors={k_neighbors}")

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
        """Ensure embeddings and citation network have same nodes."""
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

        logging.info(f"Aligned to {len(self.node_ids)} common nodes")

    def create_text_affinity_matrix(self):
        """Create affinity matrix from text embeddings using k-NN."""
        logging.info(f"Creating text affinity matrix (k-NN with k={self.k_neighbors})...")

        n_nodes = len(self.embeddings)
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(self.embeddings)

        distances, indices = nbrs.kneighbors(self.embeddings)
        similarities = 1 - distances

        # Build sparse affinity matrix
        row_indices = []
        col_indices = []
        data = []

        for i in range(n_nodes):
            for j_idx in range(1, self.k_neighbors + 1):
                neighbor = indices[i, j_idx]
                sim = similarities[i, j_idx]
                if sim > 0:
                    row_indices.extend([i, neighbor])
                    col_indices.extend([neighbor, i])
                    data.extend([sim, sim])

        self.text_affinity = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes)
        )
        self.text_affinity.setdiag(1.0)

        # Clip to valid range
        self.text_affinity.data = np.clip(self.text_affinity.data, 0.0, 1.0)

        logging.info(f"Text affinity: {self.text_affinity.shape}, nnz={self.text_affinity.nnz}")

        del nbrs, distances, indices, similarities
        gc.collect()

    def create_citation_affinity_matrix(self):
        """Create affinity matrix from citation network."""
        logging.info("Creating citation affinity matrix...")

        node_to_idx = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        n_nodes = len(self.node_ids)

        valid_edges = self.edges_df[
            (self.edges_df['source'].isin(node_to_idx.keys())) &
            (self.edges_df['target'].isin(node_to_idx.keys()))
            ]

        logging.info(f"Valid citation edges: {len(valid_edges)}")

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
        adjacency.setdiag(1.0)

        # Normalize
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1

        D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(degrees)))
        self.citation_affinity = D_inv_sqrt @ adjacency @ D_inv_sqrt

        logging.info(f"Citation affinity: {self.citation_affinity.shape}, nnz={self.citation_affinity.nnz}")

        del adjacency, degrees, D_inv_sqrt
        gc.collect()

    def perform_multiview_clustering(self):
        """Perform multiview spectral clustering."""
        logging.info("Performing multiview spectral clustering...")

        text_dense = self.text_affinity.toarray()
        citation_dense = self.citation_affinity.toarray()

        logging.info(f"View 1 (Text): shape={text_dense.shape}")
        logging.info(f"View 2 (Citation): shape={citation_dense.shape}")

        # Combine affinities (simple average)
        combined = (text_dense + citation_dense) / 2
        combined = np.clip(combined, 0.0, 1.0)
        np.fill_diagonal(combined, 1.0)
        combined = (combined + combined.T) / 2

        logging.info(f"Combined affinity - min: {combined.min():.4f}, max: {combined.max():.4f}")

        # Perform spectral clustering
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
            n_init=10,
            assign_labels='kmeans'
        )

        self.cluster_labels = sc.fit_predict(combined)

        if self.cluster_labels is None or len(self.cluster_labels) == 0:
            raise RuntimeError("Clustering failed to produce labels")

        logging.info(f"Clustering complete: {len(np.unique(self.cluster_labels))} clusters found")

        del text_dense, citation_dense, combined
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
        """Save detailed composition to CSV in same format as previous scripts."""
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
            print(f"  Rows: {len(df)}, Format: SAME as previous scripts")

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
            f"Adjusted Rand Index: {self.results['adjusted_rand_index']:.4f}\n"
            f"Normalized Mutual Info: {self.results['normalized_mutual_info']:.4f}\n"
            f"Mean Cluster Purity: {self.results['mean_cluster_purity']:.4f}\n\n"
            f"Number of Clusters: {self.results['n_clusters']}\n"
            f"Number of Papers: {self.results['n_papers']}\n"
            f"Ground Truth Classes: {self.results['n_ground_truth_classes']}"
        )
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12,
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
            print("MULTIVIEW SPECTRAL CLUSTERING PIPELINE")
            print("=" * 60)

            self.load_data()
            self.create_text_affinity_matrix()
            self.create_citation_affinity_matrix()
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

    parser = argparse.ArgumentParser(description="Multiview Spectral Clustering for Citation Networks")
    parser.add_argument("--embeddings-file", type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument("--nodes-csv", type=str, required=True, help="Path to nodes CSV file")
    parser.add_argument("--edges-csv", type=str, required=True, help="Path to edges CSV file")
    parser.add_argument("--n-clusters", type=int, default=40, help="Number of clusters")
    parser.add_argument("--k-neighbors", type=int, default=20, help="Number of neighbors for k-NN graph")
    parser.add_argument("--output-dir", type=str, default="results/multiview_clustering", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    clustering = MultiviewCitationClustering(
        embeddings_file=args.embeddings_file,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        n_clusters=args.n_clusters,
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