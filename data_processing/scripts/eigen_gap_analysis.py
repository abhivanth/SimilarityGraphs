import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.sparse.linalg import eigsh
import logging
from pathlib import Path
from typing import Tuple, List, Dict
import gc


class EfficientEigenGapAnalysis:
    """Efficient eigen gap analysis - compute eigenvalues once, find largest gap."""

    def __init__(self, embeddings: np.ndarray, max_k_analyze: int = 50):
        """
        Initialize efficient eigen gap analysis.

        Args:
            embeddings: Array of shape (n_papers, embedding_dim)
            max_k_analyze: Maximum number of clusters to analyze
        """
        self.embeddings = embeddings
        self.n_papers = len(embeddings)
        self.max_k_analyze = min(max_k_analyze, self.n_papers // 10)  # Reasonable upper limit
        self.eigenvalues = None
        self.eigen_gaps = None
        self.optimal_k = None

        logging.info(f"Initialized EFFICIENT eigen gap analysis for {self.n_papers} papers")
        logging.info(f"Will analyze up to {self.max_k_analyze} clusters")
        logging.info("Method: Single eigenvalue computation + gap analysis")

    def create_similarity_graph(self, k_neighbors: int = 20) -> sparse.csr_matrix:
        """Step 1: Build k-NN similarity graph ONCE."""
        logging.info(f"STEP 1: Creating k-NN graph with k={k_neighbors}...")

        # Use NearestNeighbors for efficient k-NN
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine', n_jobs=-1)
        nbrs.fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)

        # Convert to similarities
        similarities = 1 - distances

        # Create sparse symmetric matrix
        row_indices, col_indices, data = [], [], []

        for i in range(self.n_papers):
            # Add diagonal element first
            row_indices.append(i)
            col_indices.append(i)
            data.append(1.0)

            # Add k-NN connections
            for j_idx in range(1, k_neighbors + 1):  # Skip self
                neighbor_idx = indices[i, j_idx]
                similarity = similarities[i, j_idx]
                if similarity > 0:
                    # Make symmetric
                    row_indices.extend([i, neighbor_idx])
                    col_indices.extend([neighbor_idx, i])
                    data.extend([similarity, similarity])

        affinity_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_papers, self.n_papers)
        )

        # No need for setdiag() anymore - diagonal already included

        # Clean up
        del nbrs, distances, indices, similarities
        del row_indices, col_indices, data
        gc.collect()

        logging.info(f"✓ STEP 1 COMPLETE: Created sparse graph with {affinity_matrix.nnz} edges")
        return affinity_matrix

    def compute_graph_laplacian(self, affinity_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Step 2: Compute normalized graph Laplacian ONCE."""
        logging.info("STEP 2: Computing normalized graph Laplacian...")

        # Degree matrix
        degrees = np.array(affinity_matrix.sum(axis=1)).flatten()

        # Avoid division by zero
        degrees[degrees == 0] = 1

        # D^(-1/2)
        degrees_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        identity = sparse.eye(self.n_papers, format='csr')
        normalized_affinity = degrees_inv_sqrt @ affinity_matrix @ degrees_inv_sqrt
        laplacian = identity - normalized_affinity

        # Clean up intermediate matrices
        del degrees, degrees_inv_sqrt, identity, normalized_affinity
        gc.collect()

        logging.info("✓ STEP 2 COMPLETE: Computed normalized Laplacian")
        return laplacian

    def compute_all_eigenvalues(self, laplacian: sparse.csr_matrix) -> np.ndarray:
        """Step 3: Compute ALL eigenvalues ONCE using fast LOBPCG solver."""
        logging.info(f"STEP 3: Computing {self.max_k_analyze} smallest eigenvalues using LOBPCG solver...")

        # Compute all needed eigenvalues at once
        k_eig = min(self.max_k_analyze + 5, self.n_papers - 1)  # Add buffer for analysis

        try:
            from scipy.sparse.linalg import lobpcg

            # Create initial guess for LOBPCG - random orthogonal vectors
            np.random.seed(42)  # Reproducible results
            X = np.random.random((self.n_papers, k_eig))

            # Orthogonalize initial guess using QR decomposition
            X, _ = np.linalg.qr(X)

            logging.info(f"Using LOBPCG solver with random initial guess")

            eigenvalues, eigenvectors = lobpcg(
                laplacian,
                X,
                largest=False,  # Find smallest eigenvalues
                tol=1e-6,
                maxiter=1000,
                verbosityLevel=0
            )

            # Sort eigenvalues (should already be sorted but ensure it)
            eigenvalues = np.sort(eigenvalues)

            # Filter out negative eigenvalues (numerical errors)
            eigenvalues = eigenvalues[eigenvalues >= -1e-10]
            eigenvalues[eigenvalues < 0] = 0

            self.eigenvalues = eigenvalues
            logging.info(f"✓ STEP 3 COMPLETE: LOBPCG computed {len(eigenvalues)} eigenvalues")
            logging.info(f"Eigenvalue range: {eigenvalues[0]:.2e} to {eigenvalues[-1]:.2e}")
            logging.info(f"First 10 eigenvalues: {eigenvalues[:10]}")

            return eigenvalues

        except Exception as e:
            logging.warning(f"LOBPCG failed: {e}")
            logging.info("Falling back to ARPACK (eigsh) solver...")

            # Fallback to original eigsh method
            eigenvalues, _ = eigsh(
                laplacian,
                k=k_eig,
                which='SM',  # Smallest magnitude
                sigma=0,  # Find eigenvalues near 0
                tol=1e-6,
                maxiter=1000
            )

            # Sort eigenvalues
            eigenvalues = np.sort(eigenvalues)

            # Filter out negative eigenvalues (numerical errors)
            eigenvalues = eigenvalues[eigenvalues >= -1e-10]
            eigenvalues[eigenvalues < 0] = 0

            self.eigenvalues = eigenvalues
            logging.info(f"✓ STEP 3 COMPLETE: ARPACK computed {len(eigenvalues)} eigenvalues (fallback)")
            logging.info(f"First 10 eigenvalues: {eigenvalues[:10]}")

            return eigenvalues

    def find_largest_eigen_gap(self) -> Dict[str, any]:
        """Step 4: Find largest eigen gap to determine optimal clusters."""
        if self.eigenvalues is None:
            raise ValueError("Must compute eigenvalues first")

        logging.info("STEP 4: Analyzing eigenvalue gaps (relative) to find optimal number of clusters...")

        # Compute gaps between consecutive eigenvalues
        gaps = np.diff(self.eigenvalues)

        # Analyze up to max_k_analyze clusters
        analyze_range = min(self.max_k_analyze, len(gaps))
        gaps_to_analyze = gaps[:analyze_range]
        eigenvals_to_analyze = self.eigenvalues[:analyze_range + 1]

        # Compute relative gaps (normalize by next eigenvalue)
        relative_gaps = gaps_to_analyze / eigenvals_to_analyze[1:analyze_range + 1]

        # Find the largest RELATIVE gap (indicates optimal k)
        largest_gap_idx = np.argmax(relative_gaps)
        self.optimal_k = largest_gap_idx + 1

        # Second largest RELATIVE gap (consistent with first choice)
        relative_gaps_copy = relative_gaps.copy()
        relative_gaps_copy[largest_gap_idx] = 0  # Remove largest
        second_largest_gap_idx = np.argmax(relative_gaps_copy)
        second_optimal_k = second_largest_gap_idx + 1

        # Third largest RELATIVE gap
        relative_gaps_copy[second_largest_gap_idx] = 0
        third_largest_gap_idx = np.argmax(relative_gaps_copy)
        third_optimal_k = third_largest_gap_idx + 1

        # Additional analysis using relative gaps
        gap_ratios = []
        if len(relative_gaps) > 1:
            gap_ratios = relative_gaps[1:] / relative_gaps[:-1]

        self.eigen_gaps = gaps_to_analyze

        results = {
            'eigenvalues': eigenvals_to_analyze,
            'eigen_gaps': gaps_to_analyze,
            'relative_gaps': relative_gaps,  # Add this for transparency
            'optimal_k': self.optimal_k,
            'second_optimal_k': second_optimal_k,
            'third_optimal_k': third_optimal_k,
            'largest_gap_value': gaps_to_analyze[largest_gap_idx],
            'largest_relative_gap': relative_gaps[largest_gap_idx],  # Add this
            'largest_gap_ratio': gaps_to_analyze[largest_gap_idx] / np.mean(gaps_to_analyze),
            'gap_ratios': gap_ratios,
            'eigenvalue_analysis': {
                'n_zero_eigenvalues': np.sum(self.eigenvalues < 1e-8),
                'spectral_gap': self.eigenvalues[1] - self.eigenvalues[0] if len(self.eigenvalues) > 1 else 0,
                'eigenvalue_decay': self.eigenvalues[10] / self.eigenvalues[1] if len(self.eigenvalues) > 10 else None,
                'mean_gap': np.mean(gaps_to_analyze),
                'std_gap': np.std(gaps_to_analyze),
                'gap_significance': gaps_to_analyze[largest_gap_idx] / np.std(gaps_to_analyze),
                'relative_gap_significance': relative_gaps[largest_gap_idx] / np.std(relative_gaps)
            }
        }

        logging.info(f"✓ STEP 4 COMPLETE: Eigenvalue gap analysis finished")
        logging.info(f"🎯 OPTIMAL K (largest relative gap): {self.optimal_k}")
        logging.info(f"⭐ SECOND CHOICE: {second_optimal_k}")
        logging.info(f"   THIRD CHOICE: {third_optimal_k}")
        logging.info(f"Largest absolute gap: {gaps_to_analyze[largest_gap_idx]:.6f}")
        logging.info(f"Largest relative gap: {relative_gaps[largest_gap_idx]:.6f}")
        logging.info(f"Relative gap significance: {results['eigenvalue_analysis']['relative_gap_significance']:.2f}")

        return results

    def plot_eigenvalue_analysis(self, results: Dict, output_dir: str = "results"):
        """Create comprehensive visualization of eigenvalue analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        eigenvalues = results['eigenvalues']
        eigen_gaps = results['eigen_gaps']
        optimal_k = results['optimal_k']
        second_optimal_k = results['second_optimal_k']

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Eigenvalues with optimal points
        ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', markersize=4)
        ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.8,
                    label=f'Optimal k={optimal_k}', linewidth=2)
        ax1.axvline(x=second_optimal_k, color='orange', linestyle='--', alpha=0.6,
                    label=f'Second choice k={second_optimal_k}')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Smallest Eigenvalues of Graph Laplacian')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Eigen gaps with highlighted optimal
        gap_indices = range(1, len(eigen_gaps) + 1)
        bars = ax2.bar(gap_indices, eigen_gaps, alpha=0.7, color='lightblue')

        # Highlight optimal gaps
        bars[optimal_k - 1].set_color('red')
        bars[optimal_k - 1].set_alpha(0.9)
        if second_optimal_k <= len(bars):
            bars[second_optimal_k - 1].set_color('orange')
            bars[second_optimal_k - 1].set_alpha(0.7)

        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Eigen Gap')
        ax2.set_title('Eigen Gaps Between Consecutive Eigenvalues')
        ax2.grid(True, alpha=0.3)

        # Add text annotation for optimal gap
        ax2.text(optimal_k, eigen_gaps[optimal_k - 1],
                 f'{eigen_gaps[optimal_k - 1]:.4f}',
                 ha='center', va='bottom', fontweight='bold', color='red')

        # Plot 3: Eigenvalues (log scale)
        nonzero_eigs = eigenvalues[eigenvalues > 1e-12]
        ax3.semilogy(range(1, len(nonzero_eigs) + 1), nonzero_eigs, 'bo-', markersize=4)
        ax3.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax3.axvline(x=second_optimal_k, color='orange', linestyle='--', alpha=0.6)
        ax3.set_xlabel('Eigenvalue Index')
        ax3.set_ylabel('Eigenvalue (log scale)')
        ax3.set_title('Eigenvalues (Log Scale)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Gap analysis with statistics
        ax4.plot(gap_indices, eigen_gaps, 'go-', markersize=6, linewidth=2)
        ax4.axhline(y=np.mean(eigen_gaps), color='gray', linestyle=':', alpha=0.7, label='Mean gap')
        ax4.axhline(y=np.mean(eigen_gaps) + np.std(eigen_gaps), color='gray', linestyle='--', alpha=0.5,
                    label='Mean + 1σ')

        # Highlight top 3 gaps
        top_3_indices = np.argsort(eigen_gaps)[-3:]
        for i, idx in enumerate(top_3_indices):
            color = ['orange', 'red', 'darkred'][i]
            ax4.scatter(idx + 1, eigen_gaps[idx], s=100, color=color, zorder=5)

        ax4.set_xlabel('Number of Clusters (k)')
        ax4.set_ylabel('Eigen Gap')
        ax4.set_title('Gap Analysis with Statistical Bounds')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        plt.tight_layout()

        # Save plot
        save_path = output_path / "efficient_eigenvalue_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Eigenvalue analysis plot saved to {save_path}")

        # Create summary table
        self.create_summary_table(results, output_path)

    def create_summary_table(self, results: Dict, output_path: Path):
        """Create detailed summary table of analysis results."""
        import pandas as pd

        # Create summary data
        k_values = range(1, len(results['eigen_gaps']) + 1)
        eigenvals = results['eigenvalues'][:len(results['eigen_gaps'])]
        gaps = results['eigen_gaps']

        summary_data = {
            'k': list(k_values),
            'eigenvalue': eigenvals.tolist(),
            'eigen_gap': gaps.tolist(),
            'gap_rank': len(gaps) - np.argsort(np.argsort(gaps)),  # Rank gaps (highest = 1)
            'gap_percentile': [100 * (sum(gaps <= gap) / len(gaps)) for gap in gaps]
        }

        df = pd.DataFrame(summary_data)

        # Add optimal indicators
        df['is_optimal'] = df['k'] == results['optimal_k']
        df['is_second_choice'] = df['k'] == results['second_optimal_k']
        df['is_third_choice'] = df['k'] == results['third_optimal_k']

        # Save table
        table_path = output_path / "efficient_eigenvalue_summary.csv"
        df.to_csv(table_path, index=False)

        logging.info(f"Summary table saved to {table_path}")

        # Print comprehensive recommendations
        top_gaps = df.nlargest(10, 'eigen_gap')
        print("\n" + "=" * 60)
        print("🎯 EIGENVALUE GAP ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Method: Single eigenvalue computation + gap analysis")
        print(f"Dataset: {self.n_papers} papers, {self.embeddings.shape[1]} dimensions")
        print(f"Analyzed: {len(gaps)} cluster options")
        print()
        print("TOP 10 CLUSTER RECOMMENDATIONS (by eigen gap):")
        print("-" * 60)
        for i, (_, row) in enumerate(top_gaps.iterrows()):
            marker = "🎯" if row['is_optimal'] else "⭐" if row['is_second_choice'] else "🔸" if row[
                'is_third_choice'] else f"{i + 1:2d}"
            print(f"{marker} k={int(row['k']):2d}: gap={row['eigen_gap']:.6f} "
                  f"(rank {int(row['gap_rank'])}, {row['gap_percentile']:.1f}%ile)")

        print()
        print("ANALYSIS SUMMARY:")
        print(f"• Primary recommendation: k = {results['optimal_k']}")
        print(f"• Gap significance: {results['eigenvalue_analysis']['gap_significance']:.2f}σ")
        print(f"• Spectral gap: {results['eigenvalue_analysis']['spectral_gap']:.6f}")
        print(f"• Number of zero eigenvalues: {results['eigenvalue_analysis']['n_zero_eigenvalues']}")
        print("=" * 60)


def run_efficient_eigen_gap_analysis(embeddings_file: str,
                                     k_neighbors: int = 20,
                                     max_k_analyze: int = 50,
                                     output_dir: str = "results/efficient_eigen_gap_analysis"):
    """Run efficient eigen gap analysis pipeline - compute once, analyze gaps."""

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 60)
    logging.info("STARTING EFFICIENT EIGENVALUE GAP ANALYSIS")
    logging.info("=" * 60)

    # Load embeddings
    logging.info(f"Loading embeddings from {embeddings_file}")
    data = np.load(embeddings_file)
    embeddings = data[:, :-2]  # Remove ID and class columns

    logging.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Initialize analysis
    eigen_analysis = EfficientEigenGapAnalysis(embeddings, max_k_analyze)

    # Step 1: Create similarity graph ONCE
    affinity_matrix = eigen_analysis.create_similarity_graph(k_neighbors)

    # Step 2: Compute Laplacian ONCE
    laplacian = eigen_analysis.compute_graph_laplacian(affinity_matrix)

    # Clean up affinity matrix
    del affinity_matrix
    gc.collect()

    # Step 3: Compute ALL eigenvalues ONCE
    eigenvalues = eigen_analysis.compute_all_eigenvalues(laplacian)

    # Clean up Laplacian
    del laplacian
    gc.collect()

    # Step 4: Find largest eigen gap
    results = eigen_analysis.find_largest_eigen_gap()

    # Create visualizations
    eigen_analysis.plot_eigenvalue_analysis(results, output_dir)

    logging.info("=" * 60)
    logging.info("EFFICIENT EIGENVALUE GAP ANALYSIS COMPLETE")
    logging.info("=" * 60)

    return results, eigen_analysis.optimal_k


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Efficient eigen gap analysis for optimal cluster number")
    parser.add_argument("--embeddings-file", type=str, required=True,
                        help="Path to embeddings .npy file")
    parser.add_argument("--k-neighbors", type=int, default=20,
                        help="Number of neighbors for k-NN graph")
    parser.add_argument("--max-k-analyze", type=int, default=50,
                        help="Maximum number of clusters to analyze")
    parser.add_argument("--output-dir", type=str, default="results/efficient_eigen_gap_analysis",
                        help="Output directory")

    args = parser.parse_args()

    results, optimal_k = run_efficient_eigen_gap_analysis(
        args.embeddings_file,
        args.k_neighbors,
        args.max_k_analyze,
        args.output_dir
    )

    print(f"\n🎯 FINAL RECOMMENDATION: Use K = {optimal_k} clusters")
    print(f"📊 Results saved to: {args.output_dir}")