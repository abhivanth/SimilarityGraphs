import pandas as pd
import numpy as np
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import  cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class WeightedCitationGraphBuilder:
    """
    Build weighted citation graph by combining citation edges with embedding similarity
    """

    def __init__(self, nodes_csv_path: str, edges_csv_path: str, embeddings_npy_path: str):
        """
        Initialize builder with paths to data files

        Args:
            nodes_csv_path: Path to nodes CSV
            edges_csv_path: Path to citation edges CSV
            embeddings_npy_path: Path to embeddings NPY file [embeddings, node_id, class_idx]
        """
        self.nodes_csv_path = nodes_csv_path
        self.edges_csv_path = edges_csv_path
        self.embeddings_npy_path = embeddings_npy_path

        self.nodes_df = None
        self.edges_df = None
        self.embeddings = None
        self.node_ids = None
        self.class_labels = None
        self.node_to_embedding = {}
        self.weighted_edges_df = None

        logging.info("Initialized WeightedCitationGraphBuilder")

    def load_data(self):
        """Load all required data"""
        logging.info("Loading data...")

        # Load nodes
        self.nodes_df = pd.read_csv(self.nodes_csv_path)
        logging.info(f"Loaded {len(self.nodes_df)} nodes")
        logging.info(f"Node ID range: {self.nodes_df['node_id'].min()} - {self.nodes_df['node_id'].max()}")

        # Load edges
        self.edges_df = pd.read_csv(self.edges_csv_path)
        logging.info(f"Loaded {len(self.edges_df)} citation edges")

        # Load embeddings with labels
        embeddings_data = np.load(self.embeddings_npy_path)
        self.embeddings = embeddings_data[:, :-2]  # All columns except last two
        embedding_ids = embeddings_data[:, -2].astype(int).tolist()
        self.class_labels = embeddings_data[:, -1].astype(int)

        logging.info(f"Loaded {len(self.embeddings)} embeddings")
        logging.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        logging.info(f"Number of unique classes: {len(np.unique(self.class_labels))}")

        # Detect if embeddings use MAG IDs or node IDs
        edge_node_ids = set(self.edges_df['source'].tolist() + self.edges_df['target'].tolist())
        embedding_id_set = set(embedding_ids)
        overlap = edge_node_ids.intersection(embedding_id_set)

        if len(overlap) == 0:
            # No overlap - embeddings likely use MAG paper IDs
            logging.info("No direct overlap - embeddings appear to use MAG paper IDs")
            logging.info("Creating mapping via nodes CSV...")

            # Create mapping: mag_paper_id -> embedding
            mag_to_embedding = dict(zip(embedding_ids, self.embeddings))

            # Create final mapping: node_id -> embedding using nodes CSV
            self.node_to_embedding = {}
            mapped_count = 0
            missing_count = 0

            for _, row in self.nodes_df.iterrows():
                node_id = row['node_id']
                mag_id = row['mag_paper_id']

                if not pd.isna(mag_id) and int(mag_id) in mag_to_embedding:
                    self.node_to_embedding[node_id] = mag_to_embedding[int(mag_id)]
                    mapped_count += 1
                else:
                    missing_count += 1

            self.node_ids = list(self.node_to_embedding.keys())

            logging.info(f"Created node_id -> embedding mapping:")
            logging.info(f"  Successfully mapped: {mapped_count} nodes")
            logging.info(f"  Missing embeddings: {missing_count} nodes")

            # Verify mapping
            nodes_in_both = edge_node_ids.intersection(set(self.node_ids))
            logging.info(f"Nodes in both edges and mappings: {len(nodes_in_both)}")

            if len(nodes_in_both) == 0:
                raise ValueError(
                    "Still no overlap after MAG ID mapping! "
                    f"Edge nodes: {list(edge_node_ids)[:5]}, "
                    f"Mapped nodes: {self.node_ids[:5]}"
                )
        else:
            # Direct overlap - embeddings use node IDs
            logging.info("Direct overlap found - embeddings use node IDs")
            self.node_to_embedding = dict(zip(embedding_ids, self.embeddings))
            self.node_ids = embedding_ids

        logging.info(f"\nFinal mapping statistics:")
        logging.info(f"  Total nodes with embeddings: {len(self.node_to_embedding)}")
        logging.info(f"  Nodes in edges: {len(edge_node_ids)}")
        logging.info(f"  Nodes in both: {len(edge_node_ids.intersection(set(self.node_ids)))}")

        return self.nodes_df, self.edges_df, self.embeddings

    def compute_edge_weights(self):
        """
        Compute cosine similarity weights for each citation edge
        """
        logging.info("Computing edge weights (cosine similarity)...")

        weighted_edges = []
        missing_nodes = 0

        for idx, edge in self.edges_df.iterrows():
            source = edge['source']
            target = edge['target']

            # Get embeddings (should always exist since same dataset)
            if source in self.node_to_embedding and target in self.node_to_embedding:
                emb_source = self.node_to_embedding[source].reshape(1, -1)
                emb_target = self.node_to_embedding[target].reshape(1, -1)

                # Compute cosine similarity
                similarity = cosine_similarity(emb_source, emb_target)[0][0]

                weighted_edges.append({
                    'source': source,
                    'target': target,
                    'weight': similarity
                })
            else:
                missing_nodes += 1

        if len(weighted_edges) == 0:
            raise ValueError(
                "No valid weighted edges created! "
                "Check that node_ids in edges CSV match node_ids in embeddings NPY. "
                f"Edges checked: {len(self.edges_df)}, Missing nodes: {missing_nodes}"
            )

        self.weighted_edges_df = pd.DataFrame(weighted_edges)

        logging.info(f"Computed weights for {len(self.weighted_edges_df)} edges")
        if missing_nodes > 0:
            logging.warning(f"Skipped {missing_nodes} edges due to missing embeddings")

        # Statistics
        if len(self.weighted_edges_df) > 0:
            logging.info(f"Weight statistics:")
            logging.info(f"  Mean: {self.weighted_edges_df['weight'].mean():.4f}")
            logging.info(f"  Std: {self.weighted_edges_df['weight'].std():.4f}")
            logging.info(f"  Min: {self.weighted_edges_df['weight'].min():.4f}")
            logging.info(f"  Max: {self.weighted_edges_df['weight'].max():.4f}")

        return self.weighted_edges_df

    def create_weighted_adjacency_matrix(self, symmetric=True):
        """
        Create weighted adjacency matrix from weighted edges

        Args:
            symmetric: If True, make graph undirected by adding reverse edges

        Returns:
            Tuple of (adjacency_matrix, node_to_idx, idx_to_node)
        """
        logging.info("Creating weighted adjacency matrix...")

        # Create node mapping
        node_list = sorted(self.nodes_df['node_id'].tolist())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}

        n_nodes = len(node_list)
        row_indices = []
        col_indices = []
        data = []

        for _, edge in self.weighted_edges_df.iterrows():
            source = edge['source']
            target = edge['target']
            weight = edge['weight']

            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]

            # Add forward edge with weight
            row_indices.append(source_idx)
            col_indices.append(target_idx)
            data.append(weight)

            # Add reverse edge with SAME weight (for symmetric/undirected graph)
            if symmetric and source_idx != target_idx:
                row_indices.append(target_idx)
                col_indices.append(source_idx)
                data.append(weight)  # SAME weight

        # Create sparse weighted adjacency matrix
        adjacency_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes)
        )

        logging.info(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        logging.info(f"Non-zero entries: {adjacency_matrix.nnz}")
        logging.info(f"Matrix density: {adjacency_matrix.nnz / (n_nodes * n_nodes):.6f}")
        logging.info(f"Expected ~{2 * len(self.weighted_edges_df)} entries (2× edges for symmetry)")

        # Verify symmetry
        if symmetric:
            is_symmetric = np.allclose(
                adjacency_matrix.toarray(),
                adjacency_matrix.T.toarray(),
                atol=1e-10
            )
            logging.info(f"Matrix is symmetric: {is_symmetric}")

        return adjacency_matrix, node_to_idx, idx_to_node

    def save_weighted_edges(self, output_path: str):
        """Save weighted edges to CSV"""
        if self.weighted_edges_df is not None:
            self.weighted_edges_df.to_csv(output_path, index=False)
            logging.info(f"Saved weighted edges to {output_path}")


class HybridSpectralClusteringPipeline:
    """
    Spectral clustering pipeline for weighted citation graphs with evaluation
    """

    def __init__(self, n_clusters: int = 40, random_state: int = 42):
        """
        Initialize clustering pipeline

        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.clustering_model = None
        self.cluster_labels = None
        self.node_to_idx = None
        self.idx_to_node = None

        logging.info(f"Initialized HybridSpectralClusteringPipeline with {n_clusters} clusters")

    def fit_predict(self, weighted_adjacency_matrix, node_to_idx, idx_to_node):
        """
        Run spectral clustering on weighted adjacency matrix

        Args:
            weighted_adjacency_matrix: Sparse weighted adjacency matrix
            node_to_idx: Mapping from node_id to matrix index
            idx_to_node: Mapping from matrix index to node_id

        Returns:
            Cluster labels
        """
        logging.info("Running spectral clustering on weighted citation graph...")

        self.node_to_idx = node_to_idx
        self.idx_to_node = idx_to_node

        # Spectral clustering with precomputed weighted affinity
        self.clustering_model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            eigen_solver='lobpcg',  # Memory efficient
            assign_labels='kmeans',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=True
        )

        # THIS IS WHERE SPECTRAL CLUSTERING HAPPENS
        self.cluster_labels = self.clustering_model.fit_predict(weighted_adjacency_matrix)

        logging.info(f"Spectral clustering completed")
        logging.info(f"Found {len(np.unique(self.cluster_labels))} unique clusters")

        return self.cluster_labels

    def create_results_dataframe(self, nodes_df):
        """
        Create results dataframe with cluster assignments

        Args:
            nodes_df: Original nodes dataframe

        Returns:
            DataFrame with clustering results
        """
        # Map cluster labels back to original node IDs
        clustered_nodes = []
        for idx, cluster in enumerate(self.cluster_labels):
            original_node_id = self.idx_to_node[idx]
            clustered_nodes.append({
                'node_id': original_node_id,
                'cluster_id': cluster
            })

        clustering_df = pd.DataFrame(clustered_nodes)

        # Merge with original nodes data
        result_df = nodes_df.merge(clustering_df, on='node_id', how='inner')

        logging.info(f"Created results dataframe with {len(result_df)} nodes")

        return result_df

    def evaluate_clustering(self, result_df, embeddings, adjacency_matrix):
        """
        Evaluate clustering using ARI and Silhouette scores

        Args:
            result_df: Results dataframe with cluster assignments
            embeddings: Original embeddings for silhouette calculation
            adjacency_matrix: Weighted adjacency matrix

        Returns:
            Dictionary with evaluation metrics
        """
        logging.info("Evaluating clustering...")

        ground_truth = result_df['class_idx'].values
        predicted_clusters = result_df['cluster_id'].values

        # Adjusted Rand Index
        ari_score = adjusted_rand_score(ground_truth, predicted_clusters)
        logging.info(f"Adjusted Rand Index: {ari_score:.4f}")

        # Silhouette Score
        n_nodes = len(self.cluster_labels)

        if n_nodes > 10000:
            # Use sampling for large graphs
            logging.info("Large graph detected. Using sampling for silhouette score...")
            silhouette_avg = self._calculate_silhouette_sampling(
                self.cluster_labels, embeddings, sample_size=2000
            )
        else:
            # Use embeddings for silhouette (fair comparison with embedding-only method)
            logging.info("Computing silhouette score on embeddings...")
            silhouette_avg = silhouette_score(embeddings, self.cluster_labels)

        logging.info(f"Silhouette Score: {silhouette_avg:.4f}")

        metrics = {
            'adjusted_rand_index': float(ari_score),
            'silhouette_score': float(silhouette_avg),
            'n_clusters_true': int(len(np.unique(ground_truth))),
            'n_clusters_predicted': int(len(np.unique(predicted_clusters))),
            'n_nodes': int(len(result_df))
        }

        return metrics

    def _calculate_silhouette_sampling(self, cluster_labels, embeddings, sample_size=2000):
        """Calculate silhouette score using stratified sampling"""
        try:
            n_nodes = len(cluster_labels)
            logging.info(f"Computing silhouette via sampling ({min(sample_size, n_nodes)} of {n_nodes} nodes)...")

            if n_nodes <= sample_size:
                return silhouette_score(embeddings, cluster_labels)

            # Stratified sampling
            sample_indices = []
            unique_clusters = np.unique(cluster_labels)

            for cluster_id in unique_clusters:
                cluster_mask = cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                cluster_sample_size = max(1, int(len(cluster_indices) * sample_size / n_nodes))
                cluster_sample_size = min(cluster_sample_size, len(cluster_indices))

                if len(cluster_indices) > 0:
                    sampled = np.random.choice(
                        cluster_indices,
                        size=cluster_sample_size,
                        replace=False
                    )
                    sample_indices.extend(sampled)

            sample_indices = np.array(sample_indices[:sample_size])
            sampled_embeddings = embeddings[sample_indices]
            sampled_labels = cluster_labels[sample_indices]

            if len(np.unique(sampled_labels)) > 1:
                return silhouette_score(sampled_embeddings, sampled_labels)
            else:
                logging.warning("Only one cluster in sample")
                return None

        except Exception as e:
            logging.error(f"Silhouette calculation failed: {e}")
            return None

    def analyze_cluster_composition(self, result_df, output_dir=None):
        """
        Analyze which class labels are present in each cluster
        Same as citation network implementation

        Args:
            result_df: Results dataframe
            output_dir: Directory to save CSV

        Returns:
            Dictionary with cluster composition analysis
        """
        logging.info("Analyzing cluster composition...")

        unique_clusters = sorted(result_df['cluster_id'].unique())
        unique_classes = sorted(result_df['class_idx'].unique())

        cluster_composition = {}

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
            class_percentages = {
                class_id: (count / total_in_cluster) * 100
                for class_id, count in class_counts.items()
            }

            cluster_composition[int(cluster_id)] = {
                'total_papers': total_in_cluster,
                'class_counts': class_counts,
                'class_percentages': class_percentages,
                'num_different_classes': len(class_counts),
                'dominant_class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None,
                'dominant_class_percentage': max(class_percentages.values()) if class_percentages else 0
            }

        # Summary statistics
        summary_stats = {
            'total_clusters': len(unique_clusters),
            'total_classes': len(unique_classes),
            'avg_classes_per_cluster': np.mean([
                comp['num_different_classes'] for comp in cluster_composition.values()
            ]),
            'avg_dominant_class_percentage': np.mean([
                comp['dominant_class_percentage'] for comp in cluster_composition.values()
            ]),
            'clusters_with_single_class': sum(
                1 for comp in cluster_composition.values()
                if comp['num_different_classes'] == 1
            ),
            'clusters_with_multiple_classes': sum(
                1 for comp in cluster_composition.values()
                if comp['num_different_classes'] > 1
            )
        }

        # Print analysis
        self._print_cluster_composition(cluster_composition, summary_stats)

        # Save to CSV (same format as citation network implementation)
        if output_dir:
            self._save_cluster_composition_to_csv(cluster_composition, output_dir)

        return {
            'cluster_composition': cluster_composition,
            'summary_statistics': summary_stats
        }

    def _print_cluster_composition(self, cluster_composition, summary_stats):
        """Print detailed cluster composition analysis"""
        print("\n" + "=" * 80)
        print("HYBRID CLUSTERING - CLUSTER COMPOSITION ANALYSIS")
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
            for class_id, count in sorted_classes[:5]:
                percentage = comp['class_percentages'][class_id]
                print(f"    Class {class_id}: {count} papers ({percentage:.1f}%)")

            if len(sorted_classes) > 5:
                print(f"    ... and {len(sorted_classes) - 5} more classes")

    def _save_cluster_composition_to_csv(self, cluster_composition, output_dir):
        """
        Save detailed cluster composition to CSV
        Same format as citation_cluster_composition_detailed.csv
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

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
            csv_path = os.path.join(output_dir, "hybrid_cluster_composition_detailed.csv")
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved detailed cluster composition to {csv_path}")
            print(f"✓ Detailed cluster composition saved to {csv_path}")

        except Exception as e:
            logging.warning(f"Could not save cluster composition CSV: {e}")

    def calculate_cluster_purity(self, result_df):
        """Calculate purity for each cluster"""
        logging.info("Calculating cluster purity...")

        cluster_purities = []
        for cluster_id in result_df['cluster_id'].unique():
            cluster_data = result_df[result_df['cluster_id'] == cluster_id]
            most_common_class = cluster_data['class_idx'].mode()[0]
            purity = (cluster_data['class_idx'] == most_common_class).mean()

            cluster_purities.append({
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_data)),
                'dominant_class': int(most_common_class),
                'purity': float(purity)
            })

        purity_df = pd.DataFrame(cluster_purities)
        avg_purity = purity_df['purity'].mean()

        logging.info(f"Average cluster purity: {avg_purity:.4f}")

        return purity_df, avg_purity


class HybridClusteringRunner:
    """
    Main runner for hybrid weighted citation clustering
    """

    def __init__(self,
                 nodes_csv: str,
                 edges_csv: str,
                 embeddings_npy: str,
                 n_clusters: int = 40,
                 output_dir: str = "../results/hybrid_weighted_clustering",
                 random_state: int = 42):
        """
        Initialize runner

        Args:
            nodes_csv: Path to nodes CSV
            edges_csv: Path to edges CSV
            embeddings_npy: Path to embeddings NPY
            n_clusters: Number of clusters
            output_dir: Output directory
            random_state: Random state
        """
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.embeddings_npy = embeddings_npy
        self.n_clusters = n_clusters
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        self._setup_logging()

    def _setup_logging(self):
        """Setup logging"""
        log_file = self.output_dir / f"hybrid_clustering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def run_complete_pipeline(self):
        """Run complete hybrid clustering pipeline"""
        print("=" * 80)
        print("HYBRID WEIGHTED CITATION CLUSTERING")
        print("=" * 80)

        # Step 1: Build weighted graph
        print("\n[1/5] Building weighted citation graph...")
        builder = WeightedCitationGraphBuilder(
            self.nodes_csv,
            self.edges_csv,
            self.embeddings_npy
        )

        builder.load_data()
        builder.compute_edge_weights()

        # Save weighted edges
        weighted_edges_path = self.output_dir / "weighted_citation_edges.csv"
        builder.save_weighted_edges(str(weighted_edges_path))

        # Create weighted adjacency matrix
        adjacency_matrix, node_to_idx, idx_to_node = builder.create_weighted_adjacency_matrix(symmetric=True)

        # Step 2: Run spectral clustering
        print("\n[2/5] Running spectral clustering...")
        pipeline = HybridSpectralClusteringPipeline(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )

        cluster_labels = pipeline.fit_predict(adjacency_matrix, node_to_idx, idx_to_node)

        # Step 3: Create results dataframe
        print("\n[3/5] Creating results dataframe...")
        result_df = pipeline.create_results_dataframe(builder.nodes_df)

        # Save clustering results
        results_path = self.output_dir / "hybrid_clustering_results.csv"
        result_df.to_csv(results_path, index=False)
        logging.info(f"Saved clustering results to {results_path}")

        # Step 4: Evaluate
        print("\n[4/5] Evaluating clustering...")
        metrics = pipeline.evaluate_clustering(
            result_df,
            builder.embeddings,
            adjacency_matrix
        )

        # Calculate purity
        purity_df, avg_purity = pipeline.calculate_cluster_purity(result_df)
        metrics['average_cluster_purity'] = float(avg_purity)

        # Save purity analysis
        purity_path = self.output_dir / "hybrid_cluster_purity.csv"
        purity_df.to_csv(purity_path, index=False)

        # Step 5: Analyze composition
        print("\n[5/5] Analyzing cluster composition...")
        composition_analysis = pipeline.analyze_cluster_composition(
            result_df,
            output_dir=str(self.output_dir)
        )

        # Save metrics with composition
        metrics['composition_analysis'] = composition_analysis
        metrics_path = self.output_dir / "hybrid_clustering_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Saved metrics to {metrics_path}")

        # Print summary
        self._print_summary(metrics)

        print("\n" + "=" * 80)
        print("HYBRID CLUSTERING COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")

        return result_df, metrics, composition_analysis

    def _print_summary(self, metrics):
        """Print clustering summary"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Method: Hybrid Weighted Citation Clustering")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Number of nodes: {metrics['n_nodes']}")
        print(f"\nMetrics:")
        print(f"  Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        print(f"  Average Cluster Purity: {metrics['average_cluster_purity']:.4f}")

        comp_stats = metrics['composition_analysis']['summary_statistics']
        print(f"\nCluster Composition:")
        print(f"  Clusters with single class: {comp_stats['clusters_with_single_class']}")
        print(f"  Clusters with multiple classes: {comp_stats['clusters_with_multiple_classes']}")
        print(f"  Avg classes per cluster: {comp_stats['avg_classes_per_cluster']:.2f}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Hybrid Weighted Citation Clustering - Combine citation structure with embedding similarity"
    )

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
        help="Path to citation edges CSV file"
    )
    parser.add_argument(
        "--embeddings-npy",
        type=str,
        required=True,
        help="Path to embeddings NPY file [embeddings, node_id, class_idx]"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=40,
        help="Number of clusters (default: 40)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/hybrid_weighted_clustering",
        help="Output directory for results"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )

    args = parser.parse_args()

    # Run pipeline
    runner = HybridClusteringRunner(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        embeddings_npy=args.embeddings_npy,
        n_clusters=args.n_clusters,
        output_dir=args.output_dir,
        random_state=args.random_state
    )

    result_df, metrics, composition = runner.run_complete_pipeline()

    print("\n✓ Hybrid weighted citation clustering completed successfully!")


if __name__ == "__main__":
    main()