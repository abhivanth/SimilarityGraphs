import pandas as pd
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import warnings
import argparse
import os
import json

warnings.filterwarnings('ignore')


class CitationNetworkClustering:
    """
    Performs spectral clustering on citation networks and evaluates against ground truth
    Uses NMI (Normalized Mutual Information) instead of Silhouette Score
    """

    def __init__(self, nodes_csv_path: str, edges_csv_path: str):
        self.nodes_csv_path = nodes_csv_path
        self.edges_csv_path = edges_csv_path
        self.nodes_df = None
        self.edges_df = None
        self.graph = None
        self.adjacency_matrix = None
        self.clustering_results = {}

    def load_data(self):
        """Load nodes and edges CSV files"""
        print("Loading citation network data...")

        # Load nodes
        self.nodes_df = pd.read_csv(self.nodes_csv_path)
        print(f"Loaded {len(self.nodes_df)} nodes")
        print(f"Nodes columns: {self.nodes_df.columns.tolist()}")

        # Load edges
        self.edges_df = pd.read_csv(self.edges_csv_path)
        print(f"Loaded {len(self.edges_df)} edges")
        print(f"Edges columns: {self.edges_df.columns.tolist()}")

        # Basic statistics
        print(f"\nDataset Statistics:")
        print(f"Number of unique classes: {self.nodes_df['class_idx'].nunique()}")
        print(f"Class distribution (top 10):")
        print(self.nodes_df['class_idx'].value_counts().head(10))

        return self.nodes_df, self.edges_df

    def build_citation_graph(self):
        """Build NetworkX graph from citation edges"""
        print("\nBuilding citation graph...")

        # Create graph from edge list
        self.graph = nx.from_pandas_edgelist(
            self.edges_df,
            source='source',
            target='target',
            create_using=nx.DiGraph()  # Citation networks are directed
        )

        print(f"Graph created:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Density: {nx.density(self.graph):.6f}")

        # Check if graph is connected
        if nx.is_weakly_connected(self.graph):
            print("  Graph is weakly connected")
        else:
            print("  Graph is NOT weakly connected")
            components = list(nx.weakly_connected_components(self.graph))
            print(f"  Number of weakly connected components: {len(components)}")
            print(f"  Largest component size: {len(max(components, key=len))}")

        return self.graph

    def create_adjacency_matrix(self, symmetric=True):
        """Create adjacency matrix for spectral clustering"""
        print("\nCreating adjacency matrix...")

        # Get all nodes that appear in our dataset
        all_nodes = set(self.nodes_df['node_id'].tolist())

        # Filter edges to only include nodes in our dataset
        valid_edges = self.edges_df[
            (self.edges_df['source'].isin(all_nodes)) &
            (self.edges_df['target'].isin(all_nodes))
            ]

        print(f"Valid edges within dataset: {len(valid_edges)}")

        # Create node mapping
        node_list = sorted(all_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Create adjacency matrix
        n_nodes = len(node_list)
        row_indices = []
        col_indices = []

        for _, edge in valid_edges.iterrows():
            source_idx = node_to_idx[edge['source']]
            target_idx = node_to_idx[edge['target']]

            row_indices.append(source_idx)
            col_indices.append(target_idx)

            # Make symmetric (undirected) for spectral clustering
            if symmetric and source_idx != target_idx:
                row_indices.append(target_idx)
                col_indices.append(source_idx)

        # Create sparse matrix
        data = np.ones(len(row_indices))
        self.adjacency_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes)
        )

        self.node_mapping = node_to_idx
        self.idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        print(f"Adjacency matrix density: {self.adjacency_matrix.nnz / (n_nodes * n_nodes):.6f}")

        return self.adjacency_matrix

    def perform_spectral_clustering(self, n_clusters=None, random_state=42):
        """Perform spectral clustering on the citation graph"""
        print("\nPerforming spectral clustering...")

        if self.adjacency_matrix is None:
            self.create_adjacency_matrix()

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = self.nodes_df['class_idx'].nunique()

        print(f"Number of clusters: {n_clusters}")

        # Perform spectral clustering
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=random_state,
            n_init=10
        )

        cluster_labels = spectral.fit_predict(self.adjacency_matrix)

        # Map cluster labels back to original node IDs
        clustered_nodes = []
        for idx, cluster in enumerate(cluster_labels):
            original_node_id = self.idx_to_node[idx]
            clustered_nodes.append({
                'node_id': original_node_id,
                'cluster_id': cluster
            })

        clustering_df = pd.DataFrame(clustered_nodes)

        # Merge with original nodes data
        result_df = self.nodes_df.merge(clustering_df, on='node_id', how='inner')

        print(f"Clustering completed. {len(result_df)} nodes clustered.")
        print(f"Number of clusters found: {result_df['cluster_id'].nunique()}")

        return result_df, cluster_labels

    def calculate_evaluation_metrics(self, result_df, cluster_labels):
        """Calculate ARI and NMI scores"""
        print("\nCalculating evaluation metrics...")

        # Ground truth labels
        ground_truth = result_df['class_idx'].values
        predicted_clusters = result_df['cluster_id'].values

        # Adjusted Rand Index
        ari_score = adjusted_rand_score(ground_truth, predicted_clusters)

        # Normalized Mutual Information (NEW - replaces Silhouette)
        nmi_score = normalized_mutual_info_score(
            ground_truth,
            predicted_clusters,
            average_method='arithmetic'  # Options: 'min', 'geometric', 'arithmetic', 'max'
        )

        # Store results
        metrics = {
            'adjusted_rand_index': ari_score,
            'normalized_mutual_info': nmi_score,
            'n_clusters_true': len(np.unique(ground_truth)),
            'n_clusters_predicted': len(np.unique(predicted_clusters)),
            'n_nodes': len(result_df)
        }

        self.clustering_results = metrics

        # Print results
        print(f"\nEvaluation Results:")
        print(f"  Adjusted Rand Index: {ari_score:.4f}")
        print(f"  Normalized Mutual Information: {nmi_score:.4f}")
        print(f"  True clusters: {metrics['n_clusters_true']}")
        print(f"  Predicted clusters: {metrics['n_clusters_predicted']}")

        return metrics

    def analyze_cluster_composition(self, result_df, output_dir=None):
        """Analyze how well clusters match ground truth classes"""
        print("\nAnalyzing cluster composition...")

        # Create confusion matrix
        confusion_matrix = pd.crosstab(
            result_df['class_idx'],
            result_df['cluster_id'],
            margins=True
        )

        print("Confusion Matrix (Class vs Cluster):")
        print(confusion_matrix.head(10))

        # Calculate purity for each cluster
        cluster_purities = []
        for cluster_id in result_df['cluster_id'].unique():
            cluster_data = result_df[result_df['cluster_id'] == cluster_id]
            most_common_class = cluster_data['class_idx'].mode()[0]
            purity = (cluster_data['class_idx'] == most_common_class).mean()
            cluster_purities.append({
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'dominant_class': most_common_class,
                'purity': purity
            })

        purity_df = pd.DataFrame(cluster_purities)
        avg_purity = purity_df['purity'].mean()

        print(f"\nCluster Analysis:")
        print(f"  Average cluster purity: {avg_purity:.4f}")
        print(f"  Best cluster purity: {purity_df['purity'].max():.4f}")
        print(f"  Worst cluster purity: {purity_df['purity'].min():.4f}")

        # Detailed cluster composition analysis
        composition_analysis = self.analyze_detailed_cluster_composition(result_df)

        # Save composition analysis with output directory
        if output_dir:
            self._save_cluster_composition_to_csv(
                composition_analysis['cluster_composition'],
                output_dir
            )

        return confusion_matrix, purity_df, composition_analysis

    def analyze_detailed_cluster_composition(self, result_df):
        """
        Comprehensive analysis of which class labels are present in each cluster.

        Args:
            result_df: DataFrame with clustering results including 'cluster_id' and 'class_idx'

        Returns:
            Dictionary with detailed cluster composition analysis
        """
        print("\n" + "=" * 80)
        print("DETAILED CLUSTER COMPOSITION ANALYSIS")
        print("=" * 80)

        unique_clusters = sorted(result_df['cluster_id'].unique())
        unique_classes = sorted(result_df['class_idx'].unique())

        cluster_composition = {}

        # Create detailed composition for each cluster
        for cluster_id in unique_clusters:
            cluster_data = result_df[result_df['cluster_id'] == cluster_id]

            # Count each class in this cluster
            class_counts = {}
            for class_label in unique_classes:
                count = (cluster_data['class_idx'] == class_label).sum()
                if count > 0:
                    class_counts[int(class_label)] = int(count)

            # Calculate percentages
            total_in_cluster = len(cluster_data)
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

        return composition_analysis

    def _print_cluster_composition_analysis(self, cluster_composition: dict, summary_stats: dict):
        """Print detailed cluster composition analysis."""
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

            # Show top 5 classes in this cluster
            sorted_classes = sorted(comp['class_counts'].items(), key=lambda x: x[1], reverse=True)
            print("  Class distribution:")
            for class_id, count in sorted_classes[:5]:  # Show top 5
                percentage = comp['class_percentages'][class_id]
                print(f"    Class {class_id}: {count} papers ({percentage:.1f}%)")

            if len(sorted_classes) > 5:
                print(f"    ... and {len(sorted_classes) - 5} more classes")

    def _save_cluster_composition_to_csv(self, cluster_composition: dict, output_dir: str = None):
        """Save detailed cluster composition to CSV file."""
        try:
            if output_dir is None:
                output_dir = "../results/citation_clustering/"

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

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
            csv_path = os.path.join(output_dir, "citation_cluster_composition_detailed.csv")
            df.to_csv(csv_path, index=False)
            print(f"Detailed cluster composition saved to {csv_path}")

        except Exception as e:
            print(f"Warning: Could not save cluster composition to CSV: {e}")

    def visualize_results(self, result_df, save_path=None):
        """Create visualizations of clustering results"""
        print("\nCreating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Class distribution
        axes[0, 0].hist(result_df['class_idx'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Ground Truth Class Distribution')
        axes[0, 0].set_xlabel('Class Index')
        axes[0, 0].set_ylabel('Count')

        # 2. Cluster distribution
        axes[0, 1].hist(result_df['cluster_id'], bins=30, alpha=0.7, color='red')
        axes[0, 1].set_title('Predicted Cluster Distribution')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Count')

        # 3. Confusion matrix heatmap (subset)
        confusion_matrix = pd.crosstab(result_df['class_idx'], result_df['cluster_id'])
        # Show only top 20x20 for readability
        subset_confusion = confusion_matrix.iloc[:20, :20]
        sns.heatmap(subset_confusion, annot=False, cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix (Subset)')
        axes[1, 0].set_xlabel('Predicted Cluster')
        axes[1, 0].set_ylabel('True Class')

        # 4. Metrics summary (UPDATED - shows NMI instead of Silhouette)
        axes[1, 1].text(0.1, 0.8, f"Adjusted Rand Index: {self.clustering_results['adjusted_rand_index']:.4f}",
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.6, f"Normalized Mutual Info: {self.clustering_results['normalized_mutual_info']:.4f}",
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.4, f"True Clusters: {self.clustering_results['n_clusters_true']}",
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.1, 0.2, f"Predicted Clusters: {self.clustering_results['n_clusters_predicted']}",
                        fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Evaluation Metrics')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

        return fig

    def run_complete_analysis(self, n_clusters=None, save_results=True, output_dir="../results/"):
        """Run the complete clustering and evaluation pipeline"""
        print("=" * 60)
        print("CITATION NETWORK CLUSTERING ANALYSIS (WITH NMI)")
        print("=" * 60)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Load data
        self.load_data()

        # 2. Build graph
        self.build_citation_graph()

        # 3. Create adjacency matrix
        self.create_adjacency_matrix()

        # 4. Perform clustering
        result_df, cluster_labels = self.perform_spectral_clustering(n_clusters=n_clusters)

        # 5. Calculate metrics (ARI + NMI)
        metrics = self.calculate_evaluation_metrics(result_df, cluster_labels)

        # 6. Analyze composition
        confusion_matrix, purity_df, composition_analysis = self.analyze_cluster_composition(result_df, output_dir)

        # 7. Visualize results
        if save_results:
            viz_path = os.path.join(output_dir, "citation_clustering_analysis_nmi.png")
            self.visualize_results(result_df, save_path=viz_path)
        else:
            self.visualize_results(result_df)

        # Store results with composition analysis
        metrics['composition_analysis'] = composition_analysis

        # 8. Save results
        if save_results:
            # Save clustered data
            result_path = os.path.join(output_dir, "citation_clustering_results.csv")
            result_df.to_csv(result_path, index=False)

            # Save metrics (including composition analysis)
            metrics_path = os.path.join(output_dir, "citation_clustering_metrics_nmi.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            # Save cluster analysis
            purity_path = os.path.join(output_dir, "cluster_purity_analysis.csv")
            purity_df.to_csv(purity_path, index=False)

            print(f"\nResults saved to: {output_dir}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)

        return result_df, metrics, confusion_matrix, purity_df, composition_analysis


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Run spectral clustering on citation networks with NMI evaluation")

    parser.add_argument(
        "--nodes-csv",
        type=str,
        required=True,
        help="Path to nodes CSV file"
    )
    parser.add_argument(
        "--edges-csv",
        type=str,
        required=True,
        help="Path to edges CSV file"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: use number of unique classes)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/citation_clustering/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )

    args = parser.parse_args()

    # Initialize the clustering analyzer
    analyzer = CitationNetworkClustering(
        nodes_csv_path=args.nodes_csv,
        edges_csv_path=args.edges_csv
    )

    # Run complete analysis
    results_df, metrics, conf_matrix, purity_df, composition_analysis = analyzer.run_complete_analysis(
        n_clusters=args.n_clusters,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )

    print("\nKey Results for Comparison:")
    print(f"ARI Score: {metrics['adjusted_rand_index']:.4f}")
    print(f"NMI Score: {metrics['normalized_mutual_info']:.4f}")
    print(f"Average Cluster Purity: {purity_df['purity'].mean():.4f}")

    # Print composition summary
    comp_summary = composition_analysis['summary_statistics']
    print(f"\nCluster Composition Summary:")
    print(f"  Clusters with single class: {comp_summary['clusters_with_single_class']}")
    print(f"  Clusters with multiple classes: {comp_summary['clusters_with_multiple_classes']}")
    print(f"  Average classes per cluster: {comp_summary['avg_classes_per_cluster']:.2f}")
    print(f"  Average dominant class percentage: {comp_summary['avg_dominant_class_percentage']:.1f}%")


# Usage example (when called as script)
if __name__ == "__main__":
    main()