import pandas as pd
import os
import gzip
import numpy as np
import re
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
import arxiv


class OGBArxivLoader:
    """
    Loads OGB arXiv dataset and creates nodes CSV with titles, authors, and class labels
    """

    def __init__(self, num_nodes: int = 1000, stratified: bool = True, random_state: int = 42,
                 fetch_authors: bool = True):
        self.num_nodes = num_nodes
        self.stratified = stratified
        self.random_state = random_state
        self.fetch_authors = fetch_authors
        self.dataset = None
        self.data = None
        self.class_names = None
        self.node_to_paper_id = {}
        self.papers_df = None
        self.paper_titles = {}
        self.arxiv_client = arxiv.Client(delay_seconds=3.0) if fetch_authors else None

    def load_dataset(self):
        """Load the OGB arXiv dataset and all mappings"""
        print("Loading OGB arXiv dataset...")
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        self.data = self.dataset[0]
        print(f"Dataset loaded. Total nodes: {self.data.x.shape[0]}")
        print(f"Node features: {self.data.x.shape[1]}")
        print(f"Total edges: {self.data.edge_index.shape[1]}")

        # Load all mappings
        self.load_label_mapping()
        self.load_paper_id_mapping()
        self.load_paper_titles()

    def load_label_mapping(self):
        """Load the label to arXiv category mapping from the dataset"""
        dataset_root = self.dataset.root
        mapping_file = os.path.join(dataset_root, 'mapping', 'labelidx2arxivcategeory.csv.gz')

        print(f"Loading label mapping from: {mapping_file}")

        try:
            # Load the compressed CSV file
            with gzip.open(str(mapping_file), 'rt', encoding='utf-8') as f:
                mapping_df = pd.read_csv(f)

            # Create dictionary mapping from label index to arXiv category
            self.class_names = {}
            for idx, row in mapping_df.iterrows():
                if 'arxiv category' in mapping_df.columns:
                    category = row['arxiv category']
                elif 'category' in mapping_df.columns:
                    category = row['category']
                else:
                    # If column name is different, use the second column
                    category = row.iloc[1]

                self.class_names[idx] = category

            print(f"Loaded {len(self.class_names)} category mappings")
            print("Sample mappings:")
            for i, (idx, category) in enumerate(list(self.class_names.items())[:5]):
                print(f"  {idx}: {category}")

        except FileNotFoundError:
            print(f"Warning: Mapping file not found at {mapping_file}")
            print("Using fallback category mapping...")
            # Fallback to a simple mapping
            self.class_names = {i: f'category_{i}' for i in range(40)}
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            print("Using fallback category mapping...")
            self.class_names = {i: f'category_{i}' for i in range(40)}

    def load_paper_id_mapping(self):
        """Load the node index to MAG paper ID mapping"""
        dataset_root = self.dataset.root
        mapping_file = os.path.join(dataset_root, 'mapping', 'nodeidx2paperid.csv.gz')

        print(f"Loading paper ID mapping from: {mapping_file}")

        try:
            with gzip.open(mapping_file, 'rt') as f:
                paper_mapping_df = pd.read_csv(f)

            # Create dictionary mapping from node index to paper ID
            self.node_to_paper_id = {}
            for _, row in paper_mapping_df.iterrows():
                node_idx = row['node idx']
                paper_id = row['paper id']
                self.node_to_paper_id[node_idx] = paper_id

            print(f"Loaded {len(self.node_to_paper_id)} paper ID mappings")
            print("Sample paper ID mappings:")
            for i, (node_idx, paper_id) in enumerate(list(self.node_to_paper_id.items())[:3]):
                print(f"  Node {node_idx}: MAG Paper ID {paper_id}")

        except FileNotFoundError:
            print(f"Warning: Paper ID mapping file not found at {mapping_file}")
            self.node_to_paper_id = {}
        except Exception as e:
            print(f"Error loading paper ID mapping: {e}")
            self.node_to_paper_id = {}

    def load_paper_titles(self):
        """Load paper titles and abstracts if available"""
        dataset_root = self.dataset.root
        titleabs_file = os.path.join(dataset_root, 'titleabs.tsv')

        print(f"Looking for paper titles at: {titleabs_file}")

        try:
            if os.path.exists(titleabs_file):
                print("Loading paper titles and abstracts...")
                self.papers_df = pd.read_csv(titleabs_file, sep='\t',
                                             names=['paper_id', 'title', 'abstract'])
                print(f"Loaded {len(self.papers_df)} paper records")

                # Create quick lookup dictionary
                self.paper_titles = dict(zip(self.papers_df['paper_id'],
                                             self.papers_df['title']))
                print("Created paper title lookup dictionary")
            else:
                print("titleabs.tsv not found. Will fetch titles via API when needed.")
                self.papers_df = None
                self.paper_titles = {}

        except Exception as e:
            print(f"Error loading paper titles: {e}")
            self.papers_df = None
            self.paper_titles = {}

    def get_paper_title(self, paper_id: str) -> str:
        # check if we have it locally
        if paper_id in self.paper_titles:
            return self.paper_titles[paper_id]

        # Ultimate fallback
        return f"Paper {paper_id}"

    def clean_title_for_search(self, title: str) -> str:
        """Clean title for better arXiv API matching."""
        # Remove common LaTeX commands and symbols
        title = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', title)  # Remove \command{...}
        title = re.sub(r'\\[a-zA-Z]+', '', title)  # Remove \command
        title = re.sub(r'\$.*?\$', '', title)  # Remove $math$
        title = re.sub(r'[{}$\\]', '', title)  # Remove special chars
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        return title.strip()

    def fetch_authors_from_arxiv(self, title: str) -> str:
        """Fetch authors from arXiv API using paper title."""
        if not self.fetch_authors or not title or len(title.strip()) < 10:
            return ""

        try:
            clean_title = self.clean_title_for_search(title)
            search_query = f'ti:"{clean_title}"'

            search = arxiv.Search(
                query=search_query,
                max_results=3,  # Get top 3 matches
                sort_by=arxiv.SortCriterion.Relevance
            )

            for paper in self.arxiv_client.results(search):
                # Simple title matching - check if main words overlap
                paper_title_clean = self.clean_title_for_search(paper.title).lower()
                original_title_clean = clean_title.lower()

                # Check word overlap
                original_words = set(original_title_clean.split())
                paper_words = set(paper_title_clean.split())

                if len(original_words) > 0:
                    overlap = len(original_words.intersection(paper_words)) / len(original_words)
                    if overlap > 0.5:  # At least 50% word overlap
                        authors = [author.name for author in paper.authors]
                        return "; ".join(authors)

            return ""  # No good match found

        except Exception as e:
            print(f"Warning: Could not fetch authors for '{title[:50]}...': {e}")
            return ""

    def _get_stratified_node_indices(self):
        """Get stratified sample of node indices based on class labels."""
        if not self.stratified:
            # Original behavior: take first num_nodes
            return list(range(min(self.num_nodes, self.data.x.shape[0])))

        print(f"Creating stratified sample of {self.num_nodes} nodes...")

        # Get all node indices and their labels
        total_nodes = self.data.x.shape[0]
        all_node_indices = np.arange(total_nodes)
        labels = self.data.y.squeeze().numpy()

        # Check if we have enough nodes
        if self.num_nodes >= total_nodes:
            print(f"Requested {self.num_nodes} nodes, but dataset only has {total_nodes}. Using all nodes.")
            return list(range(total_nodes))

        try:
            # Use stratified sampling to maintain class distribution
            sampled_indices, _, sampled_labels, _ = train_test_split(
                all_node_indices,
                labels,
                train_size=self.num_nodes,
                stratify=labels,
                random_state=self.random_state
            )

            print(f"✅ Successfully created stratified sample with {len(sampled_indices)} nodes")
            return sorted(sampled_indices.tolist())  # Sort for consistency

        except ValueError as e:
            print(f"Warning: Stratified sampling failed ({e}). Using random sampling instead.")
            # Fallback to random sampling
            np.random.seed(self.random_state)
            return sorted(np.random.choice(total_nodes, size=self.num_nodes, replace=False).tolist())

    def create_nodes_csv(self, output_file: str = "ogbn_arxiv_nodes.csv"):
        """
        Create nodes CSV with paper IDs, titles, authors, and class labels
        """
        if self.data is None:
            self.load_dataset()

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Creating nodes CSV for {self.num_nodes} nodes...")

        # Get node subset - using stratified sampling
        node_indices = self._get_stratified_node_indices()

        # Get labels for these nodes
        labels = self.data.y.squeeze()

        nodes_data = []
        titles_found = 0
        authors_found = 0

        for i, node_idx in enumerate(node_indices):
            if i % 1000 == 0:
                print(f"Processing node {i + 1}/{len(node_indices)}")

            # Get label
            label_idx = labels[node_idx].item()
            class_label = self.class_names.get(label_idx, f'unknown_category_{label_idx}')

            # Get MAG paper ID
            mag_paper_id = self.node_to_paper_id.get(node_idx, None)

            # Get paper title and abstract
            title = ""
            abstract = ""
            if mag_paper_id:
                title = self.get_paper_title(str(mag_paper_id))
                paper_id = f"MAG:{mag_paper_id}"

                # Get abstract if available
                if self.papers_df is not None:
                    paper_row = self.papers_df[self.papers_df['paper_id'] == str(mag_paper_id)]
                    if not paper_row.empty:
                        abstract = str(paper_row.iloc[0]['abstract']) if pd.notna(paper_row.iloc[0]['abstract']) else ""

                # Count titles found in local data
                if not title.startswith("Paper"):
                    titles_found += 1
            else:
                title = f"Paper for Node {node_idx}"
                paper_id = f"node_{node_idx}"
                abstract = ""

            # Fetch authors from arXiv API
            authors = ""
            if self.fetch_authors and title and not title.startswith("Paper"):
                if i % 100 == 0 and i > 0:  # Show progress for author fetching
                    print(f"  Fetching authors... ({authors_found}/{i} found so far)")

                authors = self.fetch_authors_from_arxiv(title)
                if authors:
                    authors_found += 1

            nodes_data.append({
                'node_id': node_idx,
                'mag_paper_id': mag_paper_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'class_idx': label_idx,
                'class_label': class_label,
                'paper_id': paper_id
            })

        # Create DataFrame
        df = pd.DataFrame(nodes_data)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} nodes to {output_file}")

        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total nodes: {len(df)}")
        print(f"Nodes with MAG IDs: {df['mag_paper_id'].notna().sum()}")
        print(f"Titles found in local data: {titles_found}")
        print(f"Abstracts found in local data: {df['abstract'].notna().sum()}")
        print(f"Authors fetched from arXiv: {authors_found}")
        print(f"Sampling method: {'Stratified' if self.stratified else 'Sequential'}")

        print("\nClass distribution:")
        print(df['class_label'].value_counts().head(10))

        return df

    def get_edge_list(self, output_file: str = "ogbn_arxiv_edges.csv"):
        """
        Extract edge list for the subset of nodes
        """
        if self.data is None:
            self.load_dataset()

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print("Extracting citation edges...")

        # Get edges
        edge_index = self.data.edge_index

        # Filter edges to only include our subset of nodes
        if self.stratified:
            node_indices = self._get_stratified_node_indices()
            node_set = set(node_indices)
        else:
            node_set = set(range(self.num_nodes))

        edges_data = []
        for i in range(edge_index.shape[1]):
            source = edge_index[0, i].item()
            target = edge_index[1, i].item()

            if source in node_set and target in node_set:
                # Get MAG paper IDs if available
                source_mag_id = self.node_to_paper_id.get(source, None)
                target_mag_id = self.node_to_paper_id.get(target, None)

                edges_data.append({
                    'source': source,
                    'target': target,
                    'source_paper_id': f"MAG:{source_mag_id}" if source_mag_id else f"node_{source}",
                    'target_paper_id': f"MAG:{target_mag_id}" if target_mag_id else f"node_{target}"
                })

        df_edges = pd.DataFrame(edges_data)
        df_edges.to_csv(output_file, index=False)
        print(f"Saved {len(df_edges)} edges to {output_file}")

        return df_edges


# Usage example
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize loader with stratified sampling
    # fetch_authors=True will get author info from arXiv API (slower but more complete)
    # fetch_authors=False will skip author fetching (faster)
    arxiv_total_nodes = 169343
    loader = OGBArxivLoader(num_nodes=arxiv_total_nodes, stratified=True, random_state=42, fetch_authors=False)

    # Load dataset and create nodes CSV with real paper data
    print("Loading OGB arXiv dataset and creating nodes CSV...")
    nodes_df = loader.create_nodes_csv("../data/processed/ogbn_arxiv_nodes_full.csv")

    # Also create edges CSV for citation network
    print("\nCreating edges CSV...")
    edges_df = loader.get_edge_list("../data/processed/ogbn_arxiv_edges_full.csv")

    print("\n" + "=" * 60)
    print("DATASET LOADING COMPLETE!")
    print("=" * 60)
    print(f"Nodes: {len(nodes_df)}")
    print(f"Edges: {len(edges_df)}")
    print(f"Files saved to: {os.path.abspath(output_dir)}")

    # Show sample of created data
    print("\nSample nodes data:")
    print(nodes_df.head())

    print("\nSample edges data:")
    print(edges_df.head())