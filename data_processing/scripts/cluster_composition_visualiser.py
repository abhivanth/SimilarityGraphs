import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
import textwrap

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

    def create_dominant_class_bar_chart_top3(self, figsize: Tuple[int, int] = (24, 14)) -> None:
        """Create a stacked bar chart showing ONLY top 3 categories with colors based on category type."""
        print("Creating top 3 categories stacked bar chart per cluster...")

        # Collect all unique categories that appear in top 3 across all clusters
        all_top3_categories = set()
        temp_cluster_data = []
        for cluster_id in sorted(self.df['cluster_id'].unique()):
            cluster_data = self.df[self.df['cluster_id'] == cluster_id].copy()
            cluster_data = cluster_data.sort_values('paper_count', ascending=False).head(3)
            for idx, row in cluster_data.iterrows():
                if row['arxiv_category']:
                    all_top3_categories.add(row['arxiv_category'])
            temp_cluster_data.append((cluster_id, cluster_data))

        # Create a color map for all categories that appear in top 3
        unique_categories = sorted(list(all_top3_categories))
        n_categories = len(unique_categories)

        # Use a colormap with enough distinct colors
        if n_categories <= 10:
            cmap = plt.cm.tab10
        elif n_categories <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.hsv

        # Create consistent color mapping for categories
        category_colors = {}
        for i, category in enumerate(unique_categories):
            category_colors[category] = cmap(i / max(n_categories - 1, 1))

        # Add color for None/empty categories
        category_colors[None] = (0.9, 0.9, 0.9, 0.3)  # Light gray with transparency

        print(f"\nColor mapping for {n_categories} categories appearing in top 3:")
        for cat in unique_categories[:10]:  # Show first 10
            print(f"  {cat}: {ARXIV_FULL_NAMES.get(cat, cat)}")
        if n_categories > 10:
            print(f"  ... and {n_categories - 10} more categories")

        # Collect top 3 categories per cluster
        cluster_top3_data = []
        for cluster_id, cluster_data in temp_cluster_data:
            total_papers = self.df[self.df['cluster_id'] == cluster_id]['paper_count'].sum()

            top_categories = []
            for idx, row in cluster_data.iterrows():
                short_name = row['arxiv_category']
                full_name = ARXIV_FULL_NAMES.get(short_name, short_name)
                percentage = row['percentage_in_cluster']
                top_categories.append({
                    'short': short_name,
                    'full': full_name,
                    'count': row['paper_count'],
                    'percentage': percentage
                })

            # Pad with empty entries if less than 3 categories
            while len(top_categories) < 3:
                top_categories.append({
                    'short': None,
                    'full': 'None',
                    'count': 0,
                    'percentage': 0
                })

            cluster_top3_data.append({
                'cluster_id': cluster_id,
                'total_papers': total_papers,
                'top1': top_categories[0],
                'top2': top_categories[1],
                'top3': top_categories[2]
            })

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for stacked bars
        cluster_ids = [d['cluster_id'] for d in cluster_top3_data]

        # Get counts for stacking (actual paper counts) - ONLY TOP 3
        top1_counts = [d['top1']['count'] for d in cluster_top3_data]
        top2_counts = [d['top2']['count'] for d in cluster_top3_data]
        top3_counts = [d['top3']['count'] for d in cluster_top3_data]

        # Get percentages for labels
        top1_pcts = [d['top1']['percentage'] for d in cluster_top3_data]
        top2_pcts = [d['top2']['percentage'] for d in cluster_top3_data]
        top3_pcts = [d['top3']['percentage'] for d in cluster_top3_data]

        # Get category names for color assignment
        top1_categories = [d['top1']['short'] for d in cluster_top3_data]
        top2_categories = [d['top2']['short'] for d in cluster_top3_data]
        top3_categories = [d['top3']['short'] for d in cluster_top3_data]

        # Get colors for each bar segment based on category
        top1_colors = [category_colors[cat] for cat in top1_categories]
        top2_colors = [category_colors[cat] for cat in top2_categories]
        top3_colors = [category_colors[cat] for cat in top3_categories]

        # Calculate total height of stacked bars (only top 3)
        total_top3_counts = [t1 + t2 + t3 for t1, t2, t3 in zip(top1_counts, top2_counts, top3_counts)]

        # Width of bars
        bar_width = 0.8

        # Create stacked bars with category-specific colors
        # We need to draw each bar individually to assign different colors per cluster
        for i, cluster_id in enumerate(cluster_ids):
            # Bottom segment (top 1)
            if top1_counts[i] > 0:
                ax.bar(cluster_id, top1_counts[i], bar_width,
                       color=top1_colors[i], alpha=0.9, edgecolor='white', linewidth=1.5)

            # Middle segment (top 2)
            if top2_counts[i] > 0:
                ax.bar(cluster_id, top2_counts[i], bar_width,
                       bottom=top1_counts[i],
                       color=top2_colors[i], alpha=0.9, edgecolor='white', linewidth=1.5)

            # Top segment (top 3)
            if top3_counts[i] > 0:
                bottom_top3 = top1_counts[i] + top2_counts[i]
                ax.bar(cluster_id, top3_counts[i], bar_width,
                       bottom=bottom_top3,
                       color=top3_colors[i], alpha=0.9, edgecolor='white', linewidth=1.5)

        # Calculate max height for positioning labels
        max_height = max(total_top3_counts) if max(total_top3_counts) > 0 else 1

        # Threshold: bars smaller than 10% of max height get labels above
        label_inside_threshold = max_height * 0.10

        # Add text labels for top 3 categories
        for i, data in enumerate(cluster_top3_data):
            cluster_id = data['cluster_id']

            # Label for top 1 (bottom segment)
            if top1_counts[i] > 0:
                y_pos = top1_counts[i] / 2
                full_name = data['top1']['full']
                percentage = data['top1']['percentage']
                cat_color = top1_colors[i]

                if top1_counts[i] < label_inside_threshold:
                    # Place label ABOVE the bar
                    label_y = total_top3_counts[i] + max_height * 0.08
                    wrapped_text = textwrap.fill(f"1: {full_name} ({percentage:.1f}%)", width=24)
                    ax.text(cluster_id, label_y,
                            wrapped_text,
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color=cat_color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor=cat_color, linewidth=1.5))
                else:
                    # Place label inside
                    wrapped_text = textwrap.fill(f"{full_name}\n{percentage:.1f}%", width=24)
                    ax.text(cluster_id, y_pos, wrapped_text,
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white')

            # Label for top 2 (middle segment)
            if top2_counts[i] > 0:
                y_pos = top1_counts[i] + top2_counts[i] / 2
                full_name = data['top2']['full']
                percentage = data['top2']['percentage']
                cat_color = top2_colors[i]

                if top2_counts[i] < label_inside_threshold:
                    # Place label ABOVE the bar
                    label_y = total_top3_counts[i] + max_height * 0.16
                    wrapped_text = textwrap.fill(f"2: {full_name} ({percentage:.1f}%)", width=24)
                    ax.text(cluster_id, label_y,
                            wrapped_text,
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color=cat_color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor=cat_color, linewidth=1.5))
                else:
                    wrapped_text = textwrap.fill(f"{full_name}\n{percentage:.1f}%", width=24)
                    ax.text(cluster_id, y_pos, wrapped_text,
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white')

            # Label for top 3 (top segment)
            if top3_counts[i] > 0:
                y_pos = top1_counts[i] + top2_counts[i] + top3_counts[i] / 2
                full_name = data['top3']['full']
                percentage = data['top3']['percentage']
                cat_color = top3_colors[i]

                if top3_counts[i] < label_inside_threshold:
                    # Place label ABOVE the bar
                    label_y = total_top3_counts[i] + max_height * 0.24
                    wrapped_text = textwrap.fill(f"3: {full_name} ({percentage:.1f}%)", width=24)
                    ax.text(cluster_id, label_y,
                            wrapped_text,
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            color=cat_color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      alpha=0.9, edgecolor=cat_color, linewidth=1.5))
                else:
                    wrapped_text = textwrap.fill(f"{full_name}\n{percentage:.1f}%", width=24)
                    ax.text(cluster_id, y_pos, wrapped_text,
                            ha='center', va='center', fontsize=9, fontweight='bold',
                            color='white')

            # Add total paper count at the base of bar
            ax.text(cluster_id, -max_height * 0.05,
                    f'{data["total_papers"]}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='black')

        # Customize the plot
        ax.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Papers (Top 3 Categories Only)', fontsize=14, fontweight='bold')
        ax.set_title('Top 3 arXiv Categories per Cluster (Colored by Category Type)',
                     fontsize=16, fontweight='bold', pad=20)

        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Create custom legend with category colors
        # Show only the most common categories in legend
        category_frequency = {}
        for data in cluster_top3_data:
            for pos in ['top1', 'top2', 'top3']:
                cat = data[pos]['short']
                if cat:
                    category_frequency[cat] = category_frequency.get(cat, 0) + 1

        # Sort by frequency and take top 15 for legend
        top_legend_categories = sorted(category_frequency.items(),
                                       key=lambda x: x[1], reverse=True)[:15]

        legend_handles = []
        for cat, freq in top_legend_categories:
            from matplotlib.patches import Patch
            full_name = ARXIV_FULL_NAMES.get(cat, cat)
            legend_handles.append(
                Patch(facecolor=category_colors[cat], edgecolor='white',
                      label=f'{cat} ({freq}x)', alpha=0.9)
            )

        ax.legend(handles=legend_handles,
                  loc='upper left',
                  fontsize=10,
                  framealpha=0.95,
                  edgecolor='black',
                  fancybox=True,
                  title='Category (appearances)',
                  title_fontsize=11)

        # Set x-axis
        ax.set_xticks(cluster_ids)
        ax.set_xlim(min(cluster_ids) - 0.5, max(cluster_ids) + 0.5)

        # Set y-axis to accommodate labels above bars
        ax.set_ylim(-max_height * 0.08, max_height * 1.4)

        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linewidth=1)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / "top3_categories_stacked_bar_chart_by_category_color.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Top 3 categories stacked bar chart saved to: {save_path}")
        plt.close()

        # Create detailed summary table
        summary_rows = []
        for data in cluster_top3_data:
            # Calculate "other" for information purposes
            top3_total = data['top1']['count'] + data['top2']['count'] + data['top3']['count']
            other_count = data['total_papers'] - top3_total
            other_percentage = (other_count / data['total_papers'] * 100) if data['total_papers'] > 0 else 0

            summary_rows.append({
                'cluster_id': data['cluster_id'],
                'total_papers': data['total_papers'],
                'top1_category': data['top1']['short'],
                'top1_full_name': data['top1']['full'],
                'top1_count': data['top1']['count'],
                'top1_percentage': data['top1']['percentage'],
                'top2_category': data['top2']['short'],
                'top2_full_name': data['top2']['full'],
                'top2_count': data['top2']['count'],
                'top2_percentage': data['top2']['percentage'],
                'top3_category': data['top3']['short'],
                'top3_full_name': data['top3']['full'],
                'top3_count': data['top3']['count'],
                'top3_percentage': data['top3']['percentage'],
                'top3_total_count': top3_total,
                'top3_total_percentage': sum([data['top1']['percentage'],
                                              data['top2']['percentage'],
                                              data['top3']['percentage']]),
                'other_count': other_count,
                'other_percentage': other_percentage
            })

        summary_df = pd.DataFrame(summary_rows)
        csv_path = self.output_dir / "cluster_top3_categories_by_color.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Top 3 categories summary saved to: {csv_path}")

        # Also save the color mapping
        color_mapping_rows = []
        for cat in unique_categories:
            color_mapping_rows.append({
                'category': cat,
                'full_name': ARXIV_FULL_NAMES.get(cat, cat),
                'appearances_in_top3': category_frequency.get(cat, 0),
                'color_r': category_colors[cat][0],
                'color_g': category_colors[cat][1],
                'color_b': category_colors[cat][2],
            })

        color_map_df = pd.DataFrame(color_mapping_rows)
        color_map_path = self.output_dir / "category_color_mapping.csv"
        color_map_df.to_csv(color_map_path, index=False)
        print(f"Category color mapping saved to: {color_map_path}")

        # Print summary
        print(f"\nTop 3 Categories Summary (first 5 clusters):")
        for data in cluster_top3_data[:5]:
            top3_sum = data['top1']['count'] + data['top2']['count'] + data['top3']['count']
            top3_pct = sum([data['top1']['percentage'], data['top2']['percentage'], data['top3']['percentage']])

            print(
                f"\nCluster {data['cluster_id']} ({data['total_papers']} papers, top 3 = {top3_sum} papers = {top3_pct:.1f}%):")
            print(
                f"  1. {data['top1']['short']} - {data['top1']['full']}: {data['top1']['count']} ({data['top1']['percentage']:.1f}%)")
            if data['top2']['count'] > 0:
                print(
                    f"  2. {data['top2']['short']} - {data['top2']['full']}: {data['top2']['count']} ({data['top2']['percentage']:.1f}%)")
            if data['top3']['count'] > 0:
                print(
                    f"  3. {data['top3']['short']} - {data['top3']['full']}: {data['top3']['count']} ({data['top3']['percentage']:.1f}%)")

        if len(cluster_top3_data) > 5:
            print(f"\n... and {len(cluster_top3_data) - 5} more clusters")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize cluster composition with category-based colors"
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
    visualizer.create_dominant_class_bar_chart_top3()
    print("\n✅ Visualization completed with category-based coloring!")


if __name__ == "__main__":
    main()