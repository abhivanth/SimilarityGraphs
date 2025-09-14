import pandas as pd
import argparse
import os
from typing import Dict, Optional


def load_cluster_composition_csv(csv_path: str) -> pd.DataFrame:
    """Load the cluster composition CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cluster composition CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded cluster composition data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Check expected columns
    expected_cols = ['cluster_id', 'class_id', 'paper_count']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())

    return df


def load_label_mapping_csv(csv_path: str) -> pd.DataFrame:
    """Load the label index to arXiv category mapping CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label mapping CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded label mapping data: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")

    # Try different possible column names for the mapping
    possible_label_cols = ['label_idx', 'labelidx', 'class_idx', 'class_id', 'label_id', 'idx', 'label idx']
    possible_category_cols = ['arxiv_category', 'arxiv category', 'category', 'arxiv_cat', 'subject']

    label_col = None
    category_col = None

    # Find label column
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    # Find category column
    for col in possible_category_cols:
        if col in df.columns:
            category_col = col
            break

    if label_col is None:
        # Try exact match with your file's column names
        if 'label idx' in df.columns:
            label_col = 'label idx'
        else:
            raise ValueError(f"Could not find label index column. Available columns: {df.columns.tolist()}")

    if category_col is None:
        # Try exact match with your file's column names
        if 'arxiv category' in df.columns:
            category_col = 'arxiv category'
        else:
            raise ValueError(f"Could not find arxiv category column. Available columns: {df.columns.tolist()}")

    print(f"Using label column: '{label_col}' and category column: '{category_col}'")

    # Standardize column names
    df = df.rename(columns={label_col: 'class_id', category_col: 'arxiv_category'})

    # Clean up category names (remove extra whitespace, etc.)
    df['arxiv_category'] = df['arxiv_category'].astype(str).str.strip()

    return df[['class_id', 'arxiv_category']]


def merge_cluster_data_with_categories(cluster_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Merge cluster composition data with arXiv category labels."""
    print("\nMerging cluster data with arXiv categories...")

    # Ensure class_id columns are the same type
    cluster_df['class_id'] = cluster_df['class_id'].astype(int)
    mapping_df['class_id'] = mapping_df['class_id'].astype(int)

    # Merge the dataframes
    merged_df = cluster_df.merge(mapping_df, on='class_id', how='left')

    # Check for missing mappings
    missing_mappings = merged_df['arxiv_category'].isna().sum()
    if missing_mappings > 0:
        print(f"Warning: {missing_mappings} rows have missing arXiv category mappings")
        missing_class_ids = merged_df[merged_df['arxiv_category'].isna()]['class_id'].unique()
        print(f"Missing mappings for class_ids: {missing_class_ids}")

    print(f"Successfully merged data: {len(merged_df)} rows")
    return merged_df


def create_summary_statistics(merged_df: pd.DataFrame) -> Dict:
    """Create summary statistics about the cluster composition."""
    stats = {}

    # Overall statistics
    stats['total_clusters'] = merged_df['cluster_id'].nunique()
    stats['total_arxiv_categories'] = merged_df['arxiv_category'].nunique()
    stats['total_papers'] = merged_df['paper_count'].sum()

    # Category distribution
    category_counts = merged_df.groupby('arxiv_category')['paper_count'].sum().sort_values(ascending=False)
    stats['top_5_categories'] = category_counts.head().to_dict()

    # Cluster purity (dominant category percentage)
    cluster_purity = []
    for cluster_id in merged_df['cluster_id'].unique():
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id]
        total_papers = cluster_data['paper_count'].sum()
        max_papers = cluster_data['paper_count'].max()
        purity = (max_papers / total_papers) * 100 if total_papers > 0 else 0
        cluster_purity.append(purity)

    stats['avg_cluster_purity'] = sum(cluster_purity) / len(cluster_purity)
    stats['min_cluster_purity'] = min(cluster_purity)
    stats['max_cluster_purity'] = max(cluster_purity)

    return stats


def print_cluster_analysis(merged_df: pd.DataFrame):
    """Print detailed cluster analysis with arXiv categories."""
    print("\n" + "=" * 80)
    print("CLUSTER COMPOSITION WITH ARXIV CATEGORIES")
    print("=" * 80)

    for cluster_id in sorted(merged_df['cluster_id'].unique()):
        cluster_data = merged_df[merged_df['cluster_id'] == cluster_id].sort_values('paper_count', ascending=False)
        total_papers = cluster_data['paper_count'].sum()

        print(f"\nCluster {cluster_id} ({total_papers} papers):")
        print(f"  Categories present: {len(cluster_data)}")

        # Show top categories in this cluster
        print("  arXiv category distribution:")
        for _, row in cluster_data.head(5).iterrows():  # Show top 5
            percentage = (row['paper_count'] / total_papers) * 100
            print(f"    {row['arxiv_category']}: {row['paper_count']} papers ({percentage:.1f}%)")

        if len(cluster_data) > 5:
            print(f"    ... and {len(cluster_data) - 5} more categories")


def save_enhanced_csv(merged_df: pd.DataFrame, output_path: str):
    """Save the enhanced CSV with arXiv category labels."""
    # Reorder columns for better readability
    column_order = ['cluster_id', 'class_id', 'arxiv_category', 'paper_count']

    # Add any additional columns that might exist
    extra_cols = [col for col in merged_df.columns if col not in column_order]
    final_columns = column_order + extra_cols

    output_df = merged_df[final_columns]
    output_df.to_csv(output_path, index=False)
    print(f"\nEnhanced data saved to: {output_path}")


def create_category_summary_csv(merged_df: pd.DataFrame, output_dir: str):
    """Create a summary CSV grouped by arXiv category."""
    category_summary = merged_df.groupby('arxiv_category').agg({
        'paper_count': 'sum',
        'cluster_id': 'nunique'
    }).reset_index()

    category_summary.columns = ['arxiv_category', 'total_papers', 'clusters_present_in']
    category_summary['percentage_of_total'] = (category_summary['total_papers'] / category_summary[
        'total_papers'].sum()) * 100
    category_summary = category_summary.sort_values('total_papers', ascending=False)

    summary_path = os.path.join(output_dir, 'arxiv_category_summary.csv')
    category_summary.to_csv(summary_path, index=False)
    print(f"Category summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Map class IDs to arXiv category labels in cluster composition data")

    parser.add_argument(
        "--cluster-csv",
        type=str,
        required=True,
        help="Path to cluster composition CSV (cluster_id, class_id, paper_count)"
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        required=True,
        help="Path to label mapping CSV (label_idx, arxiv_category)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="Output directory for enhanced files"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="cluster_composition_with_arxiv_labels.csv",
        help="Output filename for enhanced cluster composition"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load data
        print("Loading cluster composition data...")
        cluster_df = load_cluster_composition_csv(args.cluster_csv)

        print("\nLoading label mapping data...")
        mapping_df = load_label_mapping_csv(args.mapping_csv)

        # Show available categories
        print(f"\nAvailable arXiv categories ({len(mapping_df)}):")
        for _, row in mapping_df.iterrows():
            print(f"  {row['class_id']}: {row['arxiv_category']}")

        # Merge data
        merged_df = merge_cluster_data_with_categories(cluster_df, mapping_df)

        # Create and print analysis
        print_cluster_analysis(merged_df)

        # Create summary statistics
        stats = create_summary_statistics(merged_df)
        print(f"\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total clusters: {stats['total_clusters']}")
        print(f"Total arXiv categories: {stats['total_arxiv_categories']}")
        print(f"Total papers: {stats['total_papers']}")
        print(f"Average cluster purity: {stats['avg_cluster_purity']:.1f}%")
        print(f"Top 5 categories by paper count:")
        for category, count in stats['top_5_categories'].items():
            print(f"  {category}: {count} papers")

        # Save enhanced files
        output_path = os.path.join(args.output_dir, args.output_name)
        save_enhanced_csv(merged_df, output_path)

        # Create category summary
        create_category_summary_csv(merged_df, args.output_dir)

        print(f"\nProcessing completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())