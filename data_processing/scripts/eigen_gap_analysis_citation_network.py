import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import logging
from pathlib import Path


class CitationEigenGapAnalysis:
    """Efficient eigen gap analysis for large citation networks using LOBPCG."""

    def __init__(self, nodes_file: str, edges_file: str, max_k: int = 50):
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.max_k = max_k

        # Create node mapping
        node_ids = self.nodes_df.iloc[:, 0].values  # First column as node IDs
        self.node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.n_nodes = len(node_ids)

        logging.info(f"Loaded {self.n_nodes} nodes, {len(self.edges_df)} edges")

    def create_citation_graph(self) -> sparse.csr_matrix:
        """Build undirected graph from citations."""
        logging.info("Building citation graph...")

        source_col, target_col = self.edges_df.columns[0], self.edges_df.columns[1]
        row_indices, col_indices = [], []

        # Add citation edges (make undirected)
        for _, edge in self.edges_df.iterrows():
            source_id, target_id = edge[source_col], edge[target_col]

            if source_id in self.node_to_idx and target_id in self.node_to_idx:
                i, j = self.node_to_idx[source_id], self.node_to_idx[target_id]
                if i != j:  # Avoid self-loops in citation data
                    row_indices.extend([i, j])
                    col_indices.extend([j, i])

        # Create adjacency matrix
        data = [1.0] * len(row_indices)
        adj_matrix = sparse.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_nodes, self.n_nodes)
        ).tocsr()

        # Remove duplicate edges and add self-loops
        adj_matrix.eliminate_zeros()
        adj_matrix.setdiag(1.0)

        logging.info(f"Created graph with {adj_matrix.nnz} edges")
        return adj_matrix

    def compute_laplacian(self, adj_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        """Compute normalized Laplacian efficiently."""
        logging.info("Computing normalized Laplacian...")

        degrees = np.array(adj_matrix.sum(axis=1)).flatten()
        isolated_nodes = np.sum(degrees == 0)
        if isolated_nodes > 0:
            logging.warning(f"Found {isolated_nodes} isolated nodes")
            degrees[degrees == 0] = 1  # Handle isolated nodes

        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        I = sparse.eye(self.n_nodes, format='csr')

        return I - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    def compute_eigenvalues_lobpcg(self, laplacian: sparse.csr_matrix) -> np.ndarray:
        """Compute smallest eigenvalues using efficient LOBPCG solver."""
        k_compute = min(self.max_k + 5, self.n_nodes - 1)
        logging.info(f"Computing {k_compute} smallest eigenvalues using LOBPCG...")

        # Create random orthogonal initial guess
        np.random.seed(42)  # Reproducible results
        X = np.random.random((self.n_nodes, k_compute))
        X, _ = np.linalg.qr(X)  # Orthogonalize

        try:
            eigenvals, _ = lobpcg(
                laplacian,
                X,
                largest=False,  # Find smallest eigenvalues
                tol=1e-6,
                maxiter=1000,
                verbosityLevel=0
            )

            # Sort and clean eigenvalues
            eigenvals = np.sort(eigenvals)
            eigenvals = eigenvals[eigenvals >= -1e-10]  # Remove negative numerical errors
            eigenvals[eigenvals < 0] = 0

            logging.info(f"LOBPCG computed {len(eigenvals)} eigenvalues successfully")
            logging.info(f"Eigenvalue range: {eigenvals[0]:.2e} to {eigenvals[-1]:.2e}")

            return eigenvals

        except Exception as e:
            logging.error(f"LOBPCG failed: {e}")
            logging.info("Falling back to ARPACK solver...")

            # Fallback to eigsh if LOBPCG fails
            from scipy.sparse.linalg import eigsh
            eigenvals, _ = eigsh(laplacian, k=k_compute, which='SM', sigma=0, tol=1e-6)
            eigenvals = np.sort(eigenvals)
            eigenvals = eigenvals[eigenvals >= -1e-10]
            eigenvals[eigenvals < 0] = 0

            logging.info(f"ARPACK computed {len(eigenvals)} eigenvalues (fallback)")
            return eigenvals

    def find_optimal_k(self, eigenvals: np.ndarray) -> dict:
        """Find optimal number of clusters using eigen gaps."""
        logging.info("Analyzing eigen gaps...")

        gaps = np.diff(eigenvals[:self.max_k + 1])

        # Find largest absolute gap
        optimal_k = np.argmax(gaps) + 1

        # Compute relative gaps for additional analysis
        relative_gaps = gaps / (eigenvals[1:self.max_k + 1] + 1e-10)
        optimal_k_relative = np.argmax(relative_gaps) + 1

        # Top 3 recommendations
        top_3_indices = np.argsort(gaps)[-3:][::-1]
        top_3_k = top_3_indices + 1

        return {
            'eigenvalues': eigenvals[:self.max_k + 1],
            'gaps': gaps,
            'relative_gaps': relative_gaps,
            'optimal_k': optimal_k,
            'optimal_k_relative': optimal_k_relative,
            'top_3_k': top_3_k,
            'largest_gap': gaps[optimal_k - 1],
            'n_zero_eigenvals': np.sum(eigenvals < 1e-8),
            'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0
        }

    def plot_results(self, results: dict, output_dir: str = "results"):
        """Plot eigenvalues and gaps."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        eigenvals = results['eigenvalues']
        gaps = results['gaps']
        optimal_k = results['optimal_k']
        optimal_k_relative = results['optimal_k_relative']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Eigenvalues
        ax1.plot(range(1, len(eigenvals) + 1), eigenvals, 'bo-', markersize=3)
        ax1.axvline(optimal_k, color='red', linestyle='--',
                    label=f'Optimal k={optimal_k} (abs gap)', linewidth=2)
        if optimal_k_relative != optimal_k:
            ax1.axvline(optimal_k_relative, color='green', linestyle='--',
                        label=f'k={optimal_k_relative} (rel gap)', linewidth=2)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title(f'Citation Network Eigenvalues ({self.n_nodes:,} nodes)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gaps
        bars = ax2.bar(range(1, len(gaps) + 1), gaps, alpha=0.7, color='lightblue')
        bars[optimal_k - 1].set_color('red')
        bars[optimal_k - 1].set_alpha(0.9)
        ax2.set_xlabel('k')
        ax2.set_ylabel('Gap')
        ax2.set_title('Eigenvalue Gaps')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Eigenvalues (log scale)
        nonzero_eigs = eigenvals[eigenvals > 1e-12]
        ax3.semilogy(range(1, len(nonzero_eigs) + 1), nonzero_eigs, 'go-', markersize=3)
        ax3.axvline(optimal_k, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Eigenvalue (log scale)')
        ax3.set_title('Eigenvalues (Log Scale)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Top gaps highlighted
        ax4.bar(range(1, len(gaps) + 1), gaps, alpha=0.5, color='lightgray')
        for i, k in enumerate(results['top_3_k']):
            color = ['red', 'orange', 'yellow'][i]
            bars = ax4.bar(k, gaps[k - 1], color=color, alpha=0.9,
                           label=f'#{i + 1}: k={k}')
        ax4.set_xlabel('k')
        ax4.set_ylabel('Gap')
        ax4.set_title('Top 3 Gap Recommendations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/citation_eigen_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Analysis plots saved to {output_dir}/citation_eigen_analysis.png")

    def run_analysis(self, output_dir: str = "results") -> dict:
        """Run complete eigen gap analysis pipeline."""
        # Build graph
        adj_matrix = self.create_citation_graph()

        # Compute Laplacian
        laplacian = self.compute_laplacian(adj_matrix)

        # Compute eigenvalues using LOBPCG
        eigenvals = self.compute_eigenvalues_lobpcg(laplacian)

        # Find optimal k
        results = self.find_optimal_k(eigenvals)

        # Plot results
        self.plot_results(results, output_dir)

        # Save summary
        summary_path = Path(output_dir) / "eigen_analysis_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Citation Network Eigen Gap Analysis\n")
            f.write(f"=====================================\n")
            f.write(f"Dataset: {self.n_nodes:,} nodes, {len(self.edges_df):,} citations\n")
            f.write(f"Method: LOBPCG eigenvalue solver\n\n")
            f.write(f"🎯 PRIMARY RECOMMENDATION: k = {results['optimal_k']}\n")
            f.write(f"📊 Largest gap: {results['largest_gap']:.6f}\n")
            f.write(f"📈 Alternative (relative gap): k = {results['optimal_k_relative']}\n")
            f.write(f"⭐ Top 3 recommendations: {results['top_3_k']}\n")
            f.write(f"🔍 Zero eigenvalues: {results['n_zero_eigenvals']}\n")
            f.write(f"📏 Spectral gap: {results['spectral_gap']:.6f}\n")

        # Print summary
        print(f"\n🎯 OPTIMAL K: {results['optimal_k']}")
        print(f"📊 Largest gap: {results['largest_gap']:.6f}")
        print(f"📈 Alternative: k = {results['optimal_k_relative']} (relative gap)")
        print(f"⭐ Top 3: {results['top_3_k']}")
        print(f"💾 Results saved to: {output_dir}")

        return results


def run_citation_eigen_analysis(nodes_file: str, edges_file: str,
                                max_k: int = 50, output_dir: str = "results"):
    """Run citation network eigen gap analysis with LOBPCG solver."""

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    analyzer = CitationEigenGapAnalysis(nodes_file, edges_file, max_k)
    results = analyzer.run_analysis(output_dir)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Large-scale citation network eigen gap analysis")
    parser.add_argument("--nodes", required=True, help="Path to nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to edges.csv")
    parser.add_argument("--max-k", type=int, default=50, help="Max clusters to analyze")
    parser.add_argument("--output", default="results", help="Output directory")

    args = parser.parse_args()

    print(f"🚀 Starting large-scale analysis: {args.nodes}, {args.edges}")
    results = run_citation_eigen_analysis(args.nodes, args.edges, args.max_k, args.output)