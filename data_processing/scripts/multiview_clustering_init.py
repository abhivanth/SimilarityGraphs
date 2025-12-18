from multiview_citation_clustering import MultiviewCitationClustering

# ==============================================================================
# SETUP: Define file paths
# ==============================================================================

# Path to your embeddings file
# Format: numpy array with shape (n_papers, embedding_dim + 2)
# Last two columns: [node_id, class_label]
EMBEDDINGS_FILE = "embeddings/stratified/qwen3_32b_awq_combined_embeddings_with_labels_node_ids.npy"

# Path to your citation network files
NODES_CSV = "data_processing/data/processed/ogbn_arxiv_nodes_stratified.csv"
EDGES_CSV = "data_processing/data/processed/ogbn_arxiv_edges_stratified.csv"

# ==============================================================================
# RUN MULTIVIEW CLUSTERING
# ==============================================================================

from multiview_citation_clustering import MultiviewCitationClustering

clustering = MultiviewCitationClustering(
    embeddings_file=EMBEDDINGS_FILE,
    nodes_csv=NODES_CSV,
    edges_csv=EDGES_CSV,
    n_clusters=10,
    n_spectral_dims=40,  # NEW: dimensionality of spectral embedding
    k_neighbors=5,
    output_dir="results/multiview_clustering_mvlearn",
    random_state=42
)

results = clustering.run_complete_pipeline()

# ==============================================================================
# DISPLAY RESULTS
# ==============================================================================

print("\n" + "="*70)
print("MULTIVIEW CLUSTERING RESULTS")
print("="*70)

print(f"\nOverall Metrics:")
print(f"  Adjusted Rand Index (ARI): {results['adjusted_rand_index']:.4f}")
print(f"  Normalized Mutual Info (NMI): {results['normalized_mutual_info']:.4f}")
print(f"  Mean Cluster Purity: {results['mean_cluster_purity']:.4f}")

print(f"\nDataset Info:")
print(f"  Number of papers: {results['n_papers']}")
print(f"  Number of clusters: {results['n_clusters']}")
print(f"  Ground truth classes: {results['n_ground_truth_classes']}")

print("\n" + "="*70)
print("OUTPUT FILES")
print("="*70)

print("\nThe following files have been saved:")
print("\n1. multiview_cluster_assignments.csv")
print("   - Contains: node_id, cluster_id, ground_truth_class")
print("   - Use this to see which papers are in which cluster")

print("\n2. multiview_cluster_composition_detailed.csv")
print("   - Contains: cluster_id, class_id, paper_count, percentage_in_cluster")
print("   - SAME FORMAT as your previous clustering scripts")
print("   - Use this for detailed cluster analysis")

print("\n3. multiview_metrics.json")
print("   - Contains: ARI, NMI, purity, and other metrics")
print("   - Use this for reporting results")

print("\n4. multiview_clustering_results.png")
print("   - Visualizations: distributions, confusion matrix, metrics")
print("   - Use this for presentations/papers")

print("="*70)

# ==============================================================================
# INTERPRETING RESULTS
# ==============================================================================

print("\n" + "="*70)
print("INTERPRETING THE RESULTS")
print("="*70)

print("\n1. Adjusted Rand Index (ARI):")
print("   - Range: -1 to 1 (higher is better)")
print("   - 1.0 = perfect match with ground truth")
print("   - 0.0 = random labeling")
print("   - Good scores: > 0.5")
print("   - Excellent scores: > 0.7")

print("\n2. Normalized Mutual Information (NMI):")
print("   - Range: 0 to 1 (higher is better)")
print("   - 1.0 = perfect match")
print("   - 0.0 = no mutual information")
print("   - Good scores: > 0.5")
print("   - Excellent scores: > 0.7")

print("\n3. Cluster Purity:")
print("   - Average purity across all clusters")
print("   - Measures how homogeneous each cluster is")
print("   - Higher = better")

print("="*70)

print("\n✓ Multiview clustering completed successfully!")
print(f"✓ Results saved to: results/multiview_clustering/")
print("="*70)