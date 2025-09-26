import numpy as np
import pandas as pd
import os
import gzip
import time
from typing import Dict, List, Tuple
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse
import logging


class OGBArxivKMeansAnalyzer:
    """
    K-means clustering analysis on OGB-Arxiv pre-computed 128-dimensional embeddings
    with stratified sampling and comprehensive evaluation.
    """

    def __init__(self, sample_ratio: float = 0.1, random_state: int = 42,
                 stratified_embeddings: str = None):
        """
        Initialize the K-means analyzer.

        Args:
            sample_ratio: Fraction of dataset to use (default: 0.1 for 10%)
            random_state: Random seed for reproducibility
            stratified_embeddings: Path to pre-computed stratified embeddings file (.npy)
                                 Format: [128-dim embeddings, node_id, class_label]
        """
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.stratified_embeddings = stratified_embeddings
        self.dataset = None
        self.data = None
        self.class_names = {}
        self.sampled_indices = None
        self.embeddings = None
        self.labels = None
        self.node_ids = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_dataset(self):
        """Load the OGB arXiv dataset and class mappings."""
        self.logger.info("Loading OGB arXiv dataset...")
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        self.data = self.dataset[0]

        total_nodes = self.data.x.shape[0]
        feature_dim = self.data.x.shape[1]
        total_edges = self.data.edge_index.shape[1]

        self.logger.info(f"Dataset loaded successfully:")
        self.logger.info(f"  Total nodes: {total_nodes:,}")
        self.logger.info(f"  Feature dimension: {feature_dim}")
        self.logger.info(f"  Total edges: {total_edges:,}")

        # Load class label mappings
        self._load_class_mappings()

    def _load_class_mappings(self):
        """Load the label to arXiv category mapping."""
        dataset_root = self.dataset.root
        mapping_file = os.path.join(dataset_root, 'mapping', 'labelidx2arxivcategeory.csv.gz')

        self.logger.info(f"Loading class mappings from: {mapping_file}")

        try:
            with gzip.open(mapping_file, 'rt', encoding='utf-8') as f:
                mapping_df = pd.read_csv(f)

            # Create dictionary mapping from label index to arXiv category
            self.class_names = {}
            for idx, row in mapping_df.iterrows():
                if 'arxiv category' in mapping_df.columns:
                    category = row['arxiv category']
                elif 'category' in mapping_df.columns:
                    category = row['category']
                else:
                    # Use the second column if column name is different
                    category = row.iloc[1]

                self.class_names[idx] = category

            self.logger.info(f"Loaded {len(self.class_names)} category mappings")
            self.logger.info("Sample category mappings:")
            for i, (idx, category) in enumerate(list(self.class_names.items())[:5]):
                self.logger.info(f"  {idx}: {category}")

        except FileNotFoundError:
            self.logger.warning(f"Mapping file not found. Using fallback category mapping...")
            # Fallback to simple mapping
            self.class_names = {i: f'category_{i}' for i in range(40)}
        except Exception as e:
            self.logger.warning(f"Error loading mapping file: {e}. Using fallback...")
            self.class_names = {i: f'category_{i}' for i in range(40)}

    def _load_class_mappings_standalone(self):
        """Load class mappings without requiring the full dataset to be loaded."""
        self.logger.info("Loading class mappings for pre-computed embeddings...")

        try:
            # Create a temporary dataset instance just to get the mappings
            temp_dataset = PygNodePropPredDataset(name='ogbn-arxiv')
            dataset_root = temp_dataset.root
            mapping_file = os.path.join(dataset_root, 'mapping', 'labelidx2arxivcategeory.csv.gz')

            self.logger.info(f"Loading class mappings from: {mapping_file}")

            with gzip.open(mapping_file, 'rt', encoding='utf-8') as f:
                mapping_df = pd.read_csv(f)

            # Create dictionary mapping from label index to arXiv category
            self.class_names = {}
            for idx, row in mapping_df.iterrows():
                if 'arxiv category' in mapping_df.columns:
                    category = row['arxiv category']
                elif 'category' in mapping_df.columns:
                    category = row['category']
                else:
                    # Use the second column if column name is different
                    category = row.iloc[1]

                self.class_names[idx] = category

            self.logger.info(f"Loaded {len(self.class_names)} category mappings")
            self.logger.info("Sample category mappings:")
            for i, (idx, category) in enumerate(list(self.class_names.items())[:5]):
                self.logger.info(f"  {idx}: {category}")

        except Exception as e:
            self.logger.warning(f"Could not load class mappings: {e}. Using fallback...")
            # Fallback to simple mapping
            self.class_names = {i: f'category_{i}' for i in range(40)}

    def load_stratified_embeddings(self):
        """
        Load pre-computed stratified embeddings from file.
        Expected format: [128-dim embeddings, node_id, class_label]
        """
        if not self.stratified_embeddings or not os.path.exists(self.stratified_embeddings):
            raise FileNotFoundError(f"Stratified embeddings file not found: {self.stratified_embeddings}")

        self.logger.info(f"Loading pre-computed stratified embeddings from: {self.stratified_embeddings}")

        # Load the embeddings file
        embeddings_with_labels = np.load(self.stratified_embeddings)

        self.logger.info(f"Loaded embeddings shape: {embeddings_with_labels.shape}")

        # Validate format
        if embeddings_with_labels.shape[1] != 130:  # 128 + node_id + class_label
            raise ValueError(
                f"Expected 130 columns [128-dim + node_id + class_label], got {embeddings_with_labels.shape[1]}")

        # Extract components
        self.embeddings = embeddings_with_labels[:, :128].astype(np.float32)  # First 128 dimensions
        self.node_ids = embeddings_with_labels[:, 128].astype(np.int32)  # Node IDs
        self.labels = embeddings_with_labels[:, 129].astype(np.int32)  # Class labels

        # Set sampled_indices for consistency
        self.sampled_indices = self.node_ids.tolist()

        self.logger.info(f"Extracted:")
        self.logger.info(f"  Embeddings shape: {self.embeddings.shape}")
        self.logger.info(f"  Node IDs shape: {self.node_ids.shape}")
        self.logger.info(f"  Labels shape: {self.labels.shape}")
        self.logger.info(f"  Unique classes: {len(np.unique(self.labels))}")

        # Load class mappings (needed for analysis)
        if not self.class_names:
          self._load_class_mappings_standalone()

        # Print class distribution
        self._print_class_distribution()

        return True

    def prepare_data(self):
        """
        Prepare data either by loading stratified embeddings or creating new stratified sample.
        """
        if self.stratified_embeddings:
            # Load pre-computed stratified embeddings
            self.load_stratified_embeddings()
            self.logger.info("✅ Using pre-computed stratified embeddings")
        else:
            # Create new stratified sample from OGB dataset
            self.load_dataset()
            self.create_stratified_sample()
            self.logger.info("✅ Created new stratified sample")

    def create_stratified_sample(self):
        """Create stratified sample following the exact approach from citation_network_builder."""
        if self.data is None:
            self.load_dataset()

        total_nodes = self.data.x.shape[0]
        num_sample_nodes = int(total_nodes * self.sample_ratio)

        self.logger.info(
            f"Creating stratified sample of {num_sample_nodes:,} nodes ({self.sample_ratio * 100:.1f}% of total)")

        # Get all node indices and their labels (following your exact approach)
        all_node_indices = np.arange(total_nodes)
        labels = self.data.y.squeeze().numpy()

        # Check if we have enough nodes
        if num_sample_nodes >= total_nodes:
            self.logger.info(
                f"Requested {num_sample_nodes} nodes, but dataset only has {total_nodes}. Using all nodes.")
            self.sampled_indices = list(range(total_nodes))
        else:
            try:
                # Use stratified sampling to maintain class distribution (your exact method)
                sampled_indices, _, sampled_labels, _ = train_test_split(
                    all_node_indices,
                    labels,
                    train_size=num_sample_nodes,
                    stratify=labels,
                    random_state=self.random_state
                )

                self.sampled_indices = sorted(sampled_indices.tolist())  # Sort for consistency
                self.logger.info(f"✅ Successfully created stratified sample with {len(self.sampled_indices)} nodes")

            except ValueError as e:
                self.logger.warning(f"Stratified sampling failed ({e}). Using random sampling instead.")
                # Fallback to random sampling
                np.random.seed(self.random_state)
                self.sampled_indices = sorted(
                    np.random.choice(total_nodes, size=num_sample_nodes, replace=False).tolist()
                )

        # Extract data for sampled nodes
        self.embeddings = self.data.x[self.sampled_indices].numpy()  # 128-dim pre-computed features
        self.labels = self.data.y[self.sampled_indices].squeeze().numpy()  # Class labels
        self.node_ids = np.array(self.sampled_indices)  # Node IDs

        self.logger.info(f"Extracted embeddings shape: {self.embeddings.shape}")
        self.logger.info(f"Labels shape: {self.labels.shape}")
        self.logger.info(f"Unique classes in sample: {len(np.unique(self.labels))}")

        # Print class distribution
        self._print_class_distribution()

    def _print_class_distribution(self):
        """Print class distribution in the stratified sample."""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)

        self.logger.info("\nClass distribution in stratified sample:")
        self.logger.info("-" * 60)
        self.logger.info(f"{'Class ID':<8} {'Category':<25} {'Count':<8} {'Percentage':<10}")
        self.logger.info("-" * 60)

        for class_idx in sorted(label_counts.keys()):
            count = label_counts[class_idx]
            percentage = (count / total_samples) * 100
            category_name = self.class_names.get(class_idx, f'unknown_{class_idx}')
            self.logger.info(f"{class_idx:<8} {category_name:<25} {count:<8} {percentage:<10.2f}%")

    def run_kmeans_clustering(self, k: int, n_init: int = 10):
        """
        Run K-means clustering on the pre-computed embeddings.

        Args:
            k: Number of clusters
            n_init: Number of random initializations

        Returns:
            Dictionary with clustering results and metrics
        """
        if self.embeddings is None:
            self.prepare_data()

        self.logger.info(f"\nRunning K-means clustering with k={k}")
        self.logger.info(f"Input data shape: {self.embeddings.shape}")
        self.logger.info(f"Number of classes in ground truth: {len(np.unique(self.labels))}")

        start_time = time.time()

        # Run K-means clustering
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=n_init,
            max_iter=300,
            algorithm='lloyd'  # Use standard Lloyd's algorithm
        )

        cluster_labels = kmeans.fit_predict(self.embeddings)
        clustering_time = time.time() - start_time

        self.logger.info(f"K-means clustering completed in {clustering_time:.2f} seconds")

        # Calculate evaluation metrics
        metrics = self._calculate_metrics(cluster_labels, k)

        # Analyze cluster composition
        cluster_analysis = self._analyze_cluster_composition(cluster_labels)

        results = {
            'k': k,
            'cluster_labels': cluster_labels,
            'kmeans_model': kmeans,
            'clustering_time': clustering_time,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis,
            'ground_truth_labels': self.labels,
            'node_ids': self.node_ids,
            'embeddings': self.embeddings
        }

        self._print_results_summary(results)

        return results

    def _calculate_metrics(self, cluster_labels: np.ndarray, k: int) -> Dict:
        """Calculate clustering evaluation metrics."""
        self.logger.info("Calculating evaluation metrics...")

        # Internal metrics (don't use ground truth)
        silhouette = silhouette_score(self.embeddings, cluster_labels)
        inertia = KMeans(n_clusters=k, random_state=self.random_state).fit(self.embeddings).inertia_

        # External metrics (compare with ground truth class labels)
        ari = adjusted_rand_score(self.labels, cluster_labels)
        nmi = normalized_mutual_info_score(self.labels, cluster_labels)
        homogeneity = homogeneity_score(self.labels, cluster_labels)
        completeness = completeness_score(self.labels, cluster_labels)

        metrics = {
            'silhouette_score': silhouette,
            'inertia': inertia,
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'homogeneity': homogeneity,
            'completeness': completeness
        }

        return metrics

    def _analyze_cluster_composition(self, cluster_labels: np.ndarray) -> Dict:
        """Analyze the composition of each cluster in terms of ground truth classes."""
        cluster_analysis = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            # Get indices of papers in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = self.labels[cluster_mask]
            cluster_node_ids = self.node_ids[cluster_mask]

            # Count ground truth classes in this cluster
            class_counts = Counter(cluster_true_labels)
            total_in_cluster = len(cluster_true_labels)

            # Calculate class distribution
            class_distribution = {}
            for class_idx, count in class_counts.items():
                percentage = (count / total_in_cluster) * 100
                category_name = self.class_names.get(class_idx, f'unknown_{class_idx}')
                class_distribution[class_idx] = {
                    'category_name': category_name,
                    'count': count,
                    'percentage': percentage
                }

            # Find dominant class
            dominant_class_idx = max(class_counts, key=class_counts.get)
            dominant_percentage = (class_counts[dominant_class_idx] / total_in_cluster) * 100

            cluster_analysis[cluster_id] = {
                'size': total_in_cluster,
                'dominant_class': {
                    'class_idx': dominant_class_idx,
                    'category_name': self.class_names.get(dominant_class_idx, f'unknown_{dominant_class_idx}'),
                    'count': class_counts[dominant_class_idx],
                    'percentage': dominant_percentage
                },
                'class_distribution': class_distribution,
                'diversity': len(class_counts),  # Number of different classes in cluster
                'node_ids': cluster_node_ids.tolist()
            }

        return cluster_analysis

    def _print_results_summary(self, results: Dict):
        """Print comprehensive results summary."""
        metrics = results['metrics']
        cluster_analysis = results['cluster_analysis']

        self.logger.info("\n" + "=" * 70)
        self.logger.info("K-MEANS CLUSTERING RESULTS SUMMARY")
        self.logger.info("=" * 70)

        # Basic info
        self.logger.info(f"Number of clusters (k): {results['k']}")
        self.logger.info(f"Number of papers: {len(results['cluster_labels'])}")
        self.logger.info(f"Clustering time: {results['clustering_time']:.2f} seconds")

        # Evaluation metrics
        self.logger.info(f"\nEVALUATION METRICS:")
        self.logger.info(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        self.logger.info(f"  Inertia: {metrics['inertia']:.2f}")
        self.logger.info(f"  Adjusted Rand Index (ARI): {metrics['adjusted_rand_index']:.4f}")
        self.logger.info(f"  Normalized Mutual Info (NMI): {metrics['normalized_mutual_info']:.4f}")
        self.logger.info(f"  Homogeneity: {metrics['homogeneity']:.4f}")
        self.logger.info(f"  Completeness: {metrics['completeness']:.4f}")

        # Cluster composition summary
        self.logger.info(f"\nCLUSTER COMPOSITION ANALYSIS:")
        self.logger.info("-" * 70)
        self.logger.info(f"{'Cluster':<8} {'Size':<8} {'Dominant Category':<25} {'Purity':<8} {'Diversity':<10}")
        self.logger.info("-" * 70)

        for cluster_id in sorted(cluster_analysis.keys()):
            analysis = cluster_analysis[cluster_id]
            dominant = analysis['dominant_class']

            self.logger.info(
                f"{cluster_id:<8} "
                f"{analysis['size']:<8} "
                f"{dominant['category_name'][:24]:<25} "
                f"{dominant['percentage']:<7.1f}% "
                f"{analysis['diversity']:<10}"
            )

    def save_embeddings_for_eigen_analysis(self, output_file: str = "embeddings_with_labels.npy"):
        """
        Save embeddings with node IDs and class labels for eigen gap analysis.
        Format: [128-dim embeddings, node_id, class_label]
        """
        if self.embeddings is None:
            self.prepare_data()

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Combine embeddings with node IDs and class labels
        # Shape: [n_samples, 128 + 1 + 1] = [n_samples, 130]
        node_ids_column = self.node_ids.reshape(-1, 1).astype(np.float32)
        class_labels_column = self.labels.reshape(-1, 1).astype(np.float32)

        embeddings_with_labels = np.hstack([
            self.embeddings.astype(np.float32),  # 128 dimensions
            node_ids_column,  # 1 dimension (node_id)
            class_labels_column  # 1 dimension (class_label)
        ])

        # Save to file
        np.save(output_file, embeddings_with_labels)

        self.logger.info(f"\nSaved embeddings for eigen analysis:")
        self.logger.info(f"  File: {output_file}")
        self.logger.info(f"  Shape: {embeddings_with_labels.shape}")
        self.logger.info(f"  Format: [128-dim embeddings, node_id, class_label]")
        self.logger.info(f"  Data type: {embeddings_with_labels.dtype}")

        return output_file

    def save_clustering_results(self, results: Dict, output_dir: str = "results"):
        """Save detailed clustering results to files."""
        os.makedirs(output_dir, exist_ok=True)

        k = results['k']

        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'node_id': results['node_ids'],
            'ground_truth_class': results['ground_truth_labels'],
            'ground_truth_category': [self.class_names.get(label, f'unknown_{label}')
                                      for label in results['ground_truth_labels']],
            'cluster_id': results['cluster_labels']
        })

        cluster_file = os.path.join(output_dir, f"cluster_assignments_k{k}.csv")
        cluster_df.to_csv(cluster_file, index=False)

        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df['k'] = k
        metrics_df['sample_size'] = len(results['cluster_labels'])
        metrics_df['clustering_time'] = results['clustering_time']

        metrics_file = os.path.join(output_dir, f"clustering_metrics_k{k}.csv")
        metrics_df.to_csv(metrics_file, index=False)

        # Save cluster analysis
        cluster_analysis_data = []
        for cluster_id, analysis in results['cluster_analysis'].items():
            cluster_analysis_data.append({
                'cluster_id': cluster_id,
                'size': analysis['size'],
                'dominant_class_idx': analysis['dominant_class']['class_idx'],
                'dominant_category': analysis['dominant_class']['category_name'],
                'dominant_percentage': analysis['dominant_class']['percentage'],
                'diversity': analysis['diversity']
            })

        analysis_df = pd.DataFrame(cluster_analysis_data)
        analysis_file = os.path.join(output_dir, f"cluster_analysis_k{k}.csv")
        analysis_df.to_csv(analysis_file, index=False)

        # NEW: Save cluster composition in ArXivClusterVisualizer-compatible format
        composition_file = self.save_cluster_composition_detailed(results, output_dir)

        self.logger.info(f"\nSaved clustering results to {output_dir}:")
        self.logger.info(f"  Cluster assignments: {cluster_file}")
        self.logger.info(f"  Metrics: {metrics_file}")
        self.logger.info(f"  Cluster analysis: {analysis_file}")
        self.logger.info(f"  Cluster composition (for visualization): {composition_file}")

        return {
            'cluster_assignments': cluster_file,
            'metrics': metrics_file,
            'cluster_analysis': analysis_file,
            'cluster_composition_detailed': composition_file
        }

    def save_cluster_composition_detailed(self, results: Dict, output_dir: str) -> str:
        """
        Save cluster composition in format compatible with ArXivClusterVisualizer.
        Format: cluster_id, class_id, paper_count, percentage_in_cluster
        """
        k = results['k']
        cluster_analysis = results['cluster_analysis']

        composition_data = []

        for cluster_id, analysis in cluster_analysis.items():
            total_papers_in_cluster = analysis['size']

            # Iterate through all class distributions in this cluster
            for class_idx, class_info in analysis['class_distribution'].items():
                composition_data.append({
                    'cluster_id': cluster_id,
                    'class_id': class_idx,
                    'paper_count': class_info['count'],
                    'percentage_in_cluster': class_info['percentage']
                })

        # Create DataFrame and save
        composition_df = pd.DataFrame(composition_data)

        # Sort by cluster_id and paper_count for better readability
        composition_df = composition_df.sort_values(['cluster_id', 'paper_count'], ascending=[True, False])

        composition_file = os.path.join(output_dir, f"cluster_composition_detailed_k{k}.csv")
        composition_df.to_csv(composition_file, index=False)

        self.logger.info(f"Saved detailed cluster composition with {len(composition_df)} entries")

        return composition_file


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="K-means clustering on OGB-Arxiv pre-computed embeddings")
    parser.add_argument("--k", type=int, required=True, help="Number of clusters for K-means")
    parser.add_argument("--stratified-embeddings", type=str, default=None,
                        help="Path to pre-computed stratified embeddings file (.npy). "
                             "Format: [128-dim embeddings, node_id, class_label]")
    parser.add_argument("--sample-ratio", type=float, default=0.1,
                        help="Fraction of dataset to use (default: 0.1 for 10%) - ignored if --stratified-embeddings is provided")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--embeddings-file", type=str, default="embeddings_with_labels.npy",
                        help="Output file for embeddings with labels")
    parser.add_argument("--n-init", type=int, default=10,
                        help="Number of K-means initializations")

    args = parser.parse_args()

    # Initialize analyzer with safe attribute access
    # Note: argparse converts --stratified-embeddings to stratified_embeddings (with underscores)
    analyzer = OGBArxivKMeansAnalyzer(
        sample_ratio=args.sample_ratio,
        random_state=args.random_state,
        stratified_embeddings=getattr(args, 'stratified_embeddings', None)
    )

    # Prepare data (either load stratified embeddings or create new sample)
    analyzer.prepare_data()

    # Run K-means clustering
    results = analyzer.run_kmeans_clustering(k=args.k, n_init=args.n_init)

    # Save results
    result_files = analyzer.save_clustering_results(results, args.output_dir)

    # Save embeddings for eigen analysis
    embeddings_file = analyzer.save_embeddings_for_eigen_analysis(args.embeddings_file)

    # Final summary
    print("\n" + "=" * 70)
    print("K-MEANS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"K-means with k={args.k}")
    print(f"Sample size: {len(results['cluster_labels']):,} papers ({args.sample_ratio * 100:.1f}% of total)")
    print(f"Silhouette Score: {results['metrics']['silhouette_score']:.4f}")
    print(f"Adjusted Rand Index: {results['metrics']['adjusted_rand_index']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Embeddings for eigen analysis: {embeddings_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()