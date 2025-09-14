import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# arXiv category mapping for ogbn-arxiv (40 categories) - from actual dataset
ARXIV_CATEGORY_MAP = {
    0: "cs.NA",  # Numerical Analysis
    1: "cs.MM",  # Multimedia
    2: "cs.LO",  # Logic in Computer Science
    3: "cs.CY",  # Computers and Society
    4: "cs.CR",  # Cryptography and Security
    5: "cs.DC",  # Distributed Computing
    6: "cs.HC",  # Human-Computer Interaction
    7: "cs.CE",  # Computational Engineering
    8: "cs.NI",  # Networking and Internet
    9: "cs.CC",  # Computational Complexity
    10: "cs.AI",  # Artificial Intelligence
    11: "cs.MA",  # Multiagent Systems
    12: "cs.GL",  # General Literature
    13: "cs.NE",  # Neural and Evolutionary Computing
    14: "cs.SC",  # Symbolic Computation
    15: "cs.AR",  # Hardware Architecture
    16: "cs.CV",  # Computer Vision
    17: "cs.GR",  # Graphics
    18: "cs.ET",  # Emerging Technologies
    19: "cs.SY",  # Systems and Control
    20: "cs.CG",  # Computational Geometry
    21: "cs.OH",  # Other Computer Science
    22: "cs.PL",  # Programming Languages
    23: "cs.SE",  # Software Engineering
    24: "cs.LG",  # Machine Learning
    25: "cs.SD",  # Sound
    26: "cs.SI",  # Social and Information Networks
    27: "cs.RO",  # Robotics
    28: "cs.IT",  # Information Theory
    29: "cs.PF",  # Performance
    30: "cs.CL",  # Computation and Language
    31: "cs.IR",  # Information Retrieval
    32: "cs.MS",  # Mathematical Software
    33: "cs.FL",  # Formal Languages
    34: "cs.DS",  # Data Structures and Algorithms
    35: "cs.OS",  # Operating Systems
    36: "cs.GT",  # Computer Science and Game Theory
    37: "cs.DB",  # Databases
    38: "cs.DL",  # Digital Libraries
    39: "cs.DM"  # Discrete Mathematics
}

# Full category names for visualization
ARXIV_FULL_NAMES = {
    "cs.NA": "Numerical Analysis",
    "cs.MM": "Multimedia",
    "cs.LO": "Logic in Computer Science",
    "cs.CY": "Computers and Society",
    "cs.CR": "Cryptography and Security",
    "cs.DC": "Distributed Computing",
    "cs.HC": "Human-Computer Interaction",
    "cs.CE": "Computational Engineering",
    "cs.NI": "Networking and Internet",
    "cs.CC": "Computational Complexity",
    "cs.AI": "Artificial Intelligence",
    "cs.MA": "Multiagent Systems",
    "cs.GL": "General Literature",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SC": "Symbolic Computation",
    "cs.AR": "Hardware Architecture",
    "cs.CV": "Computer Vision",
    "cs.GR": "Graphics",
    "cs.ET": "Emerging Technologies",
    "cs.SY": "Systems and Control",
    "cs.CG": "Computational Geometry",
    "cs.OH": "Other Computer Science",
    "cs.PL": "Programming Languages",
    "cs.SE": "Software Engineering",
    "cs.LG": "Machine Learning",
    "cs.SD": "Sound",
    "cs.SI": "Social and Information Networks",
    "cs.RO": "Robotics",
    "cs.IT": "Information Theory",
    "cs.PF": "Performance",
    "cs.CL": "Computation and Language",
    "cs.IR": "Information Retrieval",
    "cs.MS": "Mathematical Software",
    "cs.FL": "Formal Languages",
    "cs.DS": "Data Structures and Algorithms",
    "cs.OS": "Operating Systems",
    "cs.GT": "Computer Science and Game Theory",
    "cs.DB": "Databases",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics"
}


class ArXivClusterVisualizer:
    """Visualize cluster composition for arXiv categories."""

    def __init__(self, csv_file: str, output_dir: str = "results/arxiv_cluster_analysis",
                 mapping_file: str = None):
        """
        Initialize visualizer.

        Args:
            csv_file: Path to cluster_composition_detailed.csv
            output_dir: Output directory for visualizations
            mapping_file: Optional path to labelidx2arxivcategeory.csv for custom mapping
        """
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load custom mapping if provided
        if mapping_file:
            self.category_map = self.load_custom_mapping(mapping_file)
        else:
            self.category_map = ARXIV_CATEGORY_MAP

        # Load and process data
        self.df = self.load_and_process_data()

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def load_custom_mapping(self, mapping_file: str) -> Dict[int, str]:
        """Load category mapping from CSV file."""
        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            print(f"Warning: Mapping file not found: {mapping_file}")
            print("Using default mapping instead.")
            return ARXIV_CATEGORY_MAP

        try:
            mapping_df = pd.read_csv(mapping_file)

            # Try different possible column names
            label_col = None
            category_col = None

            for col in mapping_df.columns:
                col_lower = col.lower().replace(' ', '_')
                if 'label' in col_lower and 'idx' in col_lower:
                    label_col = col
                elif 'arxiv' in col_lower and 'category' in col_lower:
                    category_col = col

            if label_col is None or category_col is None:
                print(f"Warning: Could not identify columns in {mapping_file}")
                print(f"Available columns: {list(mapping_df.columns)}")
                print("Using default mapping instead.")
                return ARXIV_CATEGORY_MAP

            # Create mapping dictionary
            custom_map = {}
            for _, row in mapping_df.iterrows():
                idx = int(row[label_col])
                category = str(row[category_col])

                # Clean up category name (remove 'arxiv' prefix if present)
                if category.startswith('arxiv '):
                    category = category[6:]  # Remove 'arxiv '
                category = category.replace(' ', '.')  # Replace spaces with dots

                custom_map[idx] = category

            print(f"Successfully loaded {len(custom_map)} category mappings from {mapping_file}")
            return custom_map

        except Exception as e:
            print(f"Error loading mapping file {mapping_file}: {e}")
            print("Using default mapping instead.")
            return ARXIV_CATEGORY_MAP

    def load_and_process_data(self) -> pd.DataFrame:
        """Load CSV and add arXiv category names."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        df = pd.read_csv(self.csv_file)

        # Map class IDs to arXiv category names
        df['arxiv_category'] = df['class_id'].map(self.category_map)

        # Handle any unmapped categories
        unmapped = df['arxiv_category'].isna().sum()
        if unmapped > 0:
            print(f"Warning: {unmapped} unmapped class IDs found")
            # Show which IDs are unmapped
            unmapped_ids = df[df['arxiv_category'].isna()]['class_id'].unique()
            print(f"Unmapped class IDs: {unmapped_ids}")
            df['arxiv_category'] = df['arxiv_category'].fillna(
                df['class_id'].apply(lambda x: f"Unknown_{x}")
            )

        return df

    def create_cluster_heatmap(self, figsize: Tuple[int, int] = (20, 12)) -> None:
        """Create a heatmap showing paper counts for each cluster-category combination."""
        print("Creating cluster composition heatmap...")

        # Create pivot table
        pivot_data = self.df.pivot_table(
            index='arxiv_category',
            columns='cluster_id',
            values='paper_count',
            fill_value=0
        )

        # Ensure all values are integers
        pivot_data = pivot_data.astype(int)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Use a color scheme that highlights higher values
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            cbar_kws={'label': 'Number of Papers'},
            ax=ax,
            linewidths=0.5
        )

        plt.title('Cluster Composition: arXiv Categories per Cluster',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Cluster ID', fontsize=14)
        plt.ylabel('arXiv Category', fontsize=14)

        # Rotate labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)

        # Adjust layout
        plt.tight_layout()

        # Save
        save_path = self.output_dir / "cluster_composition_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        plt.close()

    def create_cluster_stacked_bars(self, max_clusters_per_plot: int = 10) -> None:
        """Create stacked bar charts showing category distribution within each cluster."""
        print("Creating cluster composition stacked bar charts...")

        clusters = sorted(self.df['cluster_id'].unique())
        n_plots = (len(clusters) + max_clusters_per_plot - 1) // max_clusters_per_plot

        for plot_idx in range(n_plots):
            start_idx = plot_idx * max_clusters_per_plot
            end_idx = min(start_idx + max_clusters_per_plot, len(clusters))
            clusters_subset = clusters[start_idx:end_idx]

            # Filter data for this subset
            subset_df = self.df[self.df['cluster_id'].isin(clusters_subset)]

            # Create pivot table for stacked bars
            pivot_data = subset_df.pivot_table(
                index='cluster_id',
                columns='arxiv_category',
                values='paper_count',
                fill_value=0
            )

            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))

            # Create stacked bar chart
            pivot_data.plot(kind='bar', stacked=True, ax=ax,
                            colormap='tab20', width=0.8)

            plt.title(f'Cluster Composition - Clusters {start_idx} to {end_idx - 1}',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Cluster ID', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)

            # Customize legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize=8, ncol=1)

            # Rotate x labels
            plt.xticks(rotation=0)

            plt.tight_layout()

            # Save
            save_path = self.output_dir / f"cluster_stacked_bars_{plot_idx + 1}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stacked bar chart {plot_idx + 1} saved to: {save_path}")
            plt.close()

    def create_category_distribution_per_cluster(self, min_papers: int = 5) -> None:
        """Create individual pie charts for clusters showing category distribution."""
        print("Creating individual cluster composition pie charts...")

        # Filter clusters with minimum papers
        cluster_sizes = self.df.groupby('cluster_id')['paper_count'].sum()
        large_clusters = cluster_sizes[cluster_sizes >= min_papers].index.tolist()

        # Determine grid size
        n_clusters = len(large_clusters)
        cols = 4
        rows = (n_clusters + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))

        # Flatten axes array for easier indexing
        if rows == 1:
            axes = [axes] if cols == 1 else axes.flatten()
        else:
            axes = axes.flatten()

        for idx, cluster_id in enumerate(large_clusters):
            ax = axes[idx]

            # Get data for this cluster
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]

            # Create pie chart
            sizes = cluster_data['paper_count'].values
            labels = cluster_data['arxiv_category'].values

            # Only show labels for categories with >5% of papers
            total_papers = sizes.sum()
            percentage_labels = [label if size / total_papers > 0.05 else ''
                                 for label, size in zip(labels, sizes)]

            ax.pie(sizes, labels=percentage_labels, autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 8})
            ax.set_title(f'Cluster {cluster_id}\n({total_papers} papers)',
                         fontweight='bold', fontsize=10)

        # Hide empty subplots
        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Category Distribution per Cluster', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = self.output_dir / "cluster_pie_charts.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie charts saved to: {save_path}")
        plt.close()

    def create_dominant_category_analysis(self) -> None:
        """Create visualization showing dominant category for each cluster."""
        print("Creating dominant category analysis...")

        # Find dominant category per cluster
        cluster_summary = []
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            dominant_idx = cluster_data['paper_count'].idxmax()
            dominant_row = cluster_data.loc[dominant_idx]

            cluster_summary.append({
                'cluster_id': cluster_id,
                'dominant_category': dominant_row['arxiv_category'],
                'dominant_count': dominant_row['paper_count'],
                'dominant_percentage': dominant_row['percentage_in_cluster'],
                'total_papers': cluster_data['paper_count'].sum(),
                'num_categories': len(cluster_data)
            })

        summary_df = pd.DataFrame(cluster_summary)

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # 1. Dominant category by cluster
        bars1 = ax1.bar(summary_df['cluster_id'], summary_df['dominant_percentage'],
                        color='skyblue', alpha=0.7)
        ax1.set_title('Dominant Category Percentage per Cluster', fontweight='bold')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Percentage (%)')
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
        ax1.legend()

        # Add value labels
        for bar, val in zip(bars1, summary_df['dominant_percentage']):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # 2. Number of categories per cluster
        bars2 = ax2.bar(summary_df['cluster_id'], summary_df['num_categories'],
                        color='lightcoral', alpha=0.7)
        ax2.set_title('Number of Categories per Cluster', fontweight='bold')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Categories')

        # 3. Total papers per cluster
        bars3 = ax3.bar(summary_df['cluster_id'], summary_df['total_papers'],
                        color='lightgreen', alpha=0.7)
        ax3.set_title('Total Papers per Cluster', fontweight='bold')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Papers')

        # 4. Dominant categories count
        category_counts = summary_df['dominant_category'].value_counts()
        ax4.barh(category_counts.index, category_counts.values, color='orange', alpha=0.7)
        ax4.set_title('How Many Clusters Each Category Dominates', fontweight='bold')
        ax4.set_xlabel('Number of Clusters Dominated')
        ax4.set_ylabel('arXiv Category')

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "dominant_category_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dominant category analysis saved to: {save_path}")
        plt.close()

        # Save summary table
        csv_path = self.output_dir / "cluster_summary_with_categories.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Cluster summary saved to: {csv_path}")

    def create_dominant_class_bar_chart(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """Create a bar chart showing dominant class for each cluster with full category names."""
        print("Creating dominant class bar chart...")

        # Find dominant category per cluster
        cluster_dominants = []
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            dominant_idx = cluster_data['paper_count'].idxmax()
            dominant_row = cluster_data.loc[dominant_idx]

            # Get full category name
            short_name = dominant_row['arxiv_category']
            full_name = ARXIV_FULL_NAMES.get(short_name, short_name)

            cluster_dominants.append({
                'cluster_id': cluster_id,
                'dominant_category_short': short_name,
                'dominant_category_full': full_name,
                'dominant_count': dominant_row['paper_count'],
                'dominant_percentage': dominant_row['percentage_in_cluster'],
                'total_papers': cluster_data['paper_count'].sum()
            })

        dominants_df = pd.DataFrame(cluster_dominants)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create color map based on category for visual grouping
        unique_categories = dominants_df['dominant_category_short'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        color_map = dict(zip(unique_categories, colors))

        bar_colors = [color_map[cat] for cat in dominants_df['dominant_category_short']]

        # Create bar chart
        bars = ax.bar(dominants_df['cluster_id'],
                      dominants_df['total_papers'],
                      color=bar_colors,
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=0.5)

        # Customize the plot
        ax.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Papers in Cluster', fontsize=14, fontweight='bold')
        ax.set_title('Cluster Sizes with Dominant arXiv Categories',
                     fontsize=16, fontweight='bold', pad=20)

        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add value labels on top of bars with full category names
        for i, (bar, row) in enumerate(zip(bars, dominants_df.itertuples())):
            # Add paper count on top
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(dominants_df['total_papers']) * 0.01,
                    f'{row.total_papers}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

            # Add category name below the bar (rotated)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.1,
                    f'{row.dominant_category_full}\n({row.dominant_percentage:.1f}%)',
                    ha='center', va='bottom',
                    rotation=90, fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Set x-axis ticks
        ax.set_xticks(dominants_df['cluster_id'])
        ax.set_xlim(-0.5, max(dominants_df['cluster_id']) + 0.5)

        # Add legend for categories (show most frequent ones)
        category_counts = dominants_df['dominant_category_short'].value_counts()
        legend_elements = []
        for cat in category_counts.index[:10]:  # Show top 10 most frequent
            full_name = ARXIV_FULL_NAMES.get(cat, cat)
            legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                                 facecolor=color_map[cat],
                                                 alpha=0.8,
                                                 label=f'{cat}: {full_name}'))

        ax.legend(handles=legend_elements,
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  fontsize=10)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "dominant_class_bar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dominant class bar chart saved to: {save_path}")
        plt.close()

        # Also create a summary table
        summary_path = self.output_dir / "cluster_dominant_categories.csv"
        dominants_df.to_csv(summary_path, index=False)
        print(f"Cluster dominant categories summary saved to: {summary_path}")

    def create_category_scatter_plot(self) -> None:
        """Create scatter plot showing cluster purity vs size."""
        print("Creating cluster purity vs size scatter plot...")

        # Calculate cluster metrics
        cluster_metrics = []
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            total_papers = cluster_data['paper_count'].sum()
            dominant_papers = cluster_data['paper_count'].max()
            dominant_category = cluster_data.loc[cluster_data['paper_count'].idxmax(), 'arxiv_category']
            purity = (dominant_papers / total_papers) * 100

            cluster_metrics.append({
                'cluster_id': cluster_id,
                'total_papers': total_papers,
                'purity': purity,
                'dominant_category': dominant_category,
                'num_categories': len(cluster_data)
            })

        metrics_df = pd.DataFrame(cluster_metrics)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by number of categories
        scatter = ax.scatter(
            metrics_df['total_papers'],
            metrics_df['purity'],
            c=metrics_df['num_categories'],
            s=100,
            alpha=0.7,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )

        # Add cluster ID labels
        for idx, row in metrics_df.iterrows():
            ax.annotate(
                f"{int(row['cluster_id'])}",
                (row['total_papers'], row['purity']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

        # Customize plot
        ax.set_xlabel('Total Papers in Cluster', fontsize=12)
        ax.set_ylabel('Cluster Purity (%)', fontsize=12)
        ax.set_title('Cluster Size vs Purity (colored by number of categories)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Categories', fontsize=12)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "cluster_purity_vs_size.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
        plt.close()

        """Create scatter plot showing cluster purity vs size."""
        print("Creating cluster purity vs size scatter plot...")

        # Calculate cluster metrics
        cluster_metrics = []
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id]
            total_papers = cluster_data['paper_count'].sum()
            dominant_papers = cluster_data['paper_count'].max()
            dominant_category = cluster_data.loc[cluster_data['paper_count'].idxmax(), 'arxiv_category']
            purity = (dominant_papers / total_papers) * 100

            cluster_metrics.append({
                'cluster_id': cluster_id,
                'total_papers': total_papers,
                'purity': purity,
                'dominant_category': dominant_category,
                'num_categories': len(cluster_data)
            })

        metrics_df = pd.DataFrame(cluster_metrics)

        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by number of categories
        scatter = ax.scatter(
            metrics_df['total_papers'],
            metrics_df['purity'],
            c=metrics_df['num_categories'],
            s=100,
            alpha=0.7,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )

        # Add cluster ID labels
        for idx, row in metrics_df.iterrows():
            ax.annotate(
                f"{int(row['cluster_id'])}",
                (row['total_papers'], row['purity']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

        # Customize plot
        ax.set_xlabel('Total Papers in Cluster', fontsize=12)
        ax.set_ylabel('Cluster Purity (%)', fontsize=12)
        ax.set_title('Cluster Size vs Purity (colored by number of categories)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Number of Categories', fontsize=12)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "cluster_purity_vs_size.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
        plt.close()

    def print_summary_stats(self) -> None:
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("ARXIV CLUSTER COMPOSITION SUMMARY")
        print("=" * 60)

        total_papers = self.df['paper_count'].sum()
        n_clusters = self.df['cluster_id'].nunique()
        n_categories = self.df['class_id'].nunique()

        print(f"Total papers: {total_papers:,}")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of arXiv categories: {n_categories}")

        # Cluster size statistics
        cluster_sizes = self.df.groupby('cluster_id')['paper_count'].sum()
        print(f"\nCluster size statistics:")
        print(f"  Mean: {cluster_sizes.mean():.1f}")
        print(f"  Median: {cluster_sizes.median():.1f}")
        print(f"  Min: {cluster_sizes.min()}")
        print(f"  Max: {cluster_sizes.max()}")

        # Category distribution
        print(f"\nMost common categories overall:")
        category_totals = self.df.groupby('arxiv_category')['paper_count'].sum().sort_values(ascending=False)
        for category, count in category_totals.head(10).items():
            percentage = (count / total_papers) * 100
            print(f"  {category}: {count:,} papers ({percentage:.1f}%)")

        print("=" * 60)

    def run_all_visualizations(self) -> None:
        """Run all visualization methods."""
        print(f"Starting arXiv cluster visualization analysis...")
        print(f"Input file: {self.csv_file}")
        print(f"Output directory: {self.output_dir}")

        # Print summary statistics
        self.print_summary_stats()

        # Create all visualizations
        self.create_cluster_heatmap()
        self.create_cluster_stacked_bars()
        self.create_category_distribution_per_cluster()
        self.create_dominant_category_analysis()
        self.create_dominant_class_bar_chart()  # NEW: Clean bar chart with full names
        self.create_category_scatter_plot()

        print(f"\n✅ All visualizations completed!")
        print(f"Check the output directory: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize cluster composition for arXiv categories"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        required=True,
        help="Path to cluster_composition_detailed.csv file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/arxiv_cluster_analysis",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--mapping-file",
        type=str,
        help="Path to labelidx2arxivcategeory.csv file for custom category mapping"
    )

    args = parser.parse_args()

    # Create and run visualizer
    visualizer = ArXivClusterVisualizer(
        args.csv_file,
        args.output_dir,
        args.mapping_file
    )
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()