import pandas as pd
import requests
import time
import gzip
from typing import Dict
from ogb.nodeproppred import PygNodePropPredDataset


class OGBArxivLoader:
    """
    Loads OGB arXiv dataset and creates nodes CSV with titles and class labels
    """

    def __init__(self, num_nodes: int = 1000):
        self.num_nodes = num_nodes
        self.dataset = None
        self.data = None
        self.class_names = None
        self.node_to_paper_id = {}
        self.papers_df = None
        self.paper_titles = {}

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
        mapping_file = os.path.join(dataset_root, 'mapping', 'labelidx2arxivcategory.csv.gz')

        print(f"Loading label mapping from: {mapping_file}")

        try:
            # Load the compressed CSV file
            with gzip.open(mapping_file, 'rt') as f:
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
        """Get paper title from local data or API"""
        # First check if we have it locally
        if paper_id in self.paper_titles:
            return self.paper_titles[paper_id]

        # If not, try to fetch from Semantic Scholar API
        title = self.fetch_title_from_semantic_scholar(paper_id)
        if title and title != "Unknown Title":
            # Cache it for future use
            self.paper_titles[paper_id] = title
            return title

        # Fallback to MAG API
        title = self.fetch_title_from_mag_api(paper_id)
        if title and title != "Unknown Title":
            self.paper_titles[paper_id] = title
            return title

        # Ultimate fallback
        return f"Paper {paper_id}"

    def fetch_title_from_semantic_scholar(self, mag_paper_id: str) -> str:
        """Fetch paper title from Semantic Scholar API using MAG ID"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/MAG:{mag_paper_id}"
        headers = {
            'User-Agent': 'Academic Research Tool (contact: researcher@university.edu)'
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title', 'Unknown Title')
                if title and title.strip():
                    return title.strip()
        except Exception as e:
            print(f"Error fetching from Semantic Scholar for MAG:{mag_paper_id}: {e}")

        return "Unknown Title"

    def fetch_title_from_mag_api(self, mag_paper_id: str) -> str:
        """Fetch paper title from Microsoft Academic Graph API"""
        # Note: MAG API was discontinued, but keeping this as template
        # for other potential APIs
        try:
            # This is a placeholder - MAG API is no longer available
            # You could implement other academic APIs here
            return "Unknown Title"
        except Exception as e:
            print(f"Error fetching from MAG API for {mag_paper_id}: {e}")
            return "Unknown Title"

    def fetch_arxiv_metadata(self, arxiv_id: str) -> Dict:
        """
        Fetch paper metadata from arXiv API
        """
        # Clean the arXiv ID if it's in the format "node_xxx"
        if arxiv_id.startswith("node_"):
            # For demonstration, we'll create dummy data
            # In practice, you'd need actual arXiv IDs
            return {
                'title': f'Paper Title for {arxiv_id}',
                'abstract': f'Abstract for {arxiv_id}',
                'categories': 'cs.LG'
            }

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Parse XML response (simplified)
                # In practice, you'd use xml.etree.ElementTree
                return {
                    'title': 'Fetched Title',
                    'abstract': 'Fetched Abstract',
                    'categories': 'cs.LG'
                }
        except Exception as e:
            print(f"Error fetching {arxiv_id}: {e}")

        return {
            'title': f'Unknown Title for {arxiv_id}',
            'abstract': 'Unknown Abstract',
            'categories': 'cs.OH'
        }

    def fetch_semantic_scholar_metadata(self, paper_id: str) -> Dict:
        """
        Alternative: Fetch metadata from Semantic Scholar API
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; research-bot/1.0)'
        }

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', 'Unknown Title'),
                    'abstract': data.get('abstract', 'Unknown Abstract'),
                    'categories': 'cs.LG'  # Default category
                }
        except Exception as e:
            print(f"Error fetching from Semantic Scholar {paper_id}: {e}")

        return {
            'title': f'Unknown Title for {paper_id}',
            'abstract': 'Unknown Abstract',
            'categories': 'cs.OH'
        }

    def create_nodes_csv(self, output_file: str = "ogbn_arxiv_nodes.csv"):
        """
        Create nodes CSV with paper IDs, titles, and class labels
        """
        if self.data is None:
            self.load_dataset()

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Creating nodes CSV for {self.num_nodes} nodes...")

        # Get node subset
        node_indices = list(range(min(self.num_nodes, self.data.x.shape[0])))

        # Get labels for these nodes
        labels = self.data.y.squeeze()

        nodes_data = []

        for i, node_idx in enumerate(node_indices):
            if i % 50 == 0:
                print(f"Processing node {i + 1}/{len(node_indices)}")

            # Get label
            label_idx = labels[node_idx].item()
            class_label = self.class_names.get(label_idx, f'unknown_category_{label_idx}')

            # Get MAG paper ID
            mag_paper_id = self.node_to_paper_id.get(node_idx, None)

            # Get paper title
            if mag_paper_id:
                title = self.get_paper_title(str(mag_paper_id))
                paper_id = f"MAG:{mag_paper_id}"
            else:
                title = f"Paper for Node {node_idx}"
                paper_id = f"node_{node_idx}"

            # Get publication year if available
            year = None
            if hasattr(self.data, 'node_year') and self.data.node_year is not None:
                year = self.data.node_year[node_idx].item()

            nodes_data.append({
                'node_id': node_idx,
                'mag_paper_id': mag_paper_id,
                'title': title,
               'class_idx': label_idx,
            })

            # Rate limiting for API calls
            if mag_paper_id and str(mag_paper_id) not in self.paper_titles:
                time.sleep(0.1)  # Be respectful to APIs

        # Create DataFrame
        df = pd.DataFrame(nodes_data)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} nodes to {output_file}")

        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total nodes: {len(df)}")
        print(f"Nodes with MAG IDs: {df['mag_paper_id'].notna().sum()}")
        print(f"Nodes with fetched titles: {df['title'].str.startswith('Paper for Node').sum() == 0}")

        print("\nClass distribution:")
        #print(df['class_label'].value_counts().head(10))

        # if year:
        #     print(f"\nYear range: {df['publication_year'].min()} - {df['publication_year'].max()}")

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
    import os

    # Create output directory if it doesn't exist
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize loader
    loader = OGBArxivLoader(num_nodes=1000)

    # Load dataset and create nodes CSV with real paper data
    print("Loading OGB arXiv dataset and creating nodes CSV...")
    nodes_df = loader.create_nodes_csv("../data/processed/ogbn_arxiv_nodes.csv")

    # Also create edges CSV for citation network
    print("\nCreating edges CSV...")
    edges_df = loader.get_edge_list("../data/processed/ogbn_arxiv_edges.csv")

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