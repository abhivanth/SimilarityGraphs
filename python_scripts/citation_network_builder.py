import gzip
import tarfile
import networkx as nx
import re
from typing import Dict, List, Tuple
import pandas as pd
from pathlib import Path


class CitationNetworkParser:
    """Parse cit-HepTh dataset and export citation network."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.citations_file = self.data_dir / "cit-HepTh.txt.gz"
        self.abstracts_file = self.data_dir / "cit-HepTh-abstracts.tar.gz"
        self.graph = nx.DiGraph()
        self.paper_metadata = {}
        
    def verify_files(self) -> None:
        """Verify required data files exist."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.citations_file.exists():
            raise FileNotFoundError(
                f"Citations file not found: {self.citations_file}\n"
                f"Please download 'cit-HepTh.txt.gz' and place it in {self.data_dir}"
            )
        
        if not self.abstracts_file.exists():
            raise FileNotFoundError(
                f"Abstracts file not found: {self.abstracts_file}\n"
                f"Please download 'cit-HepTh-abstracts.tar.gz' and place it in {self.data_dir}"
            )
        
        print(f"✓ Found citations file: {self.citations_file}")
        print(f"✓ Found abstracts file: {self.abstracts_file}")
    
    def parse_citations(self) -> List[Tuple[str, str]]:
        """Parse citation edges from compressed file."""
        edges = []
        
        print("Parsing citations...")
        with gzip.open(self.citations_file, 'rt') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        edges.append((parts[0], parts[1]))
        
        print(f"Found {len(edges)} citations")
        return edges
    
    def parse_abstracts(self) -> None:
        """Extract paper metadata from tar.gz archive."""
        print("Extracting paper metadata...")
        
        with tarfile.open(self.abstracts_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.abs'):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode('utf-8', errors='ignore')
                        metadata = self._extract_metadata(content)
                        
                        if 'paper_id' in metadata:
                            self.paper_metadata[metadata['paper_id']] = metadata
        
        print(f"Extracted metadata for {len(self.paper_metadata)} papers")
    
    def _extract_metadata(self, content: str) -> Dict[str, str]:
        """Extract metadata fields from paper abstract file."""
        metadata = {}
        
        # Extract paper ID
        paper_id_match = re.search(r'Paper:\s*(\S+)', content)
        if paper_id_match:
            metadata['paper_id'] = paper_id_match.group(1)
        
        # Extract title
        title_match = re.search(r'Title:\s*(.+?)(?=\nAuthors?:|$)', content, re.DOTALL)
        if title_match:
            title = ' '.join(title_match.group(1).strip().split())
            metadata['title'] = title
        
        # Extract authors
        authors_match = re.search(r'Authors?:\s*(.+?)(?=\nComments?:|Abstract:|$)', content, re.DOTALL)
        if authors_match:
            authors = ' '.join(authors_match.group(1).strip().split())
            metadata['authors'] = authors
        
        # Extract abstract
        abstract_match = re.search(r'Abstract:\s*(.+?)(?=\\\\|$)', content, re.DOTALL)
        if abstract_match:
            abstract = ' '.join(abstract_match.group(1).strip().split())
            metadata['abstract'] = abstract
        
        return metadata
    
    def build_graph(self, edges: List[Tuple[str, str]]) -> None:
        """Build NetworkX graph with metadata."""
        print("Building graph...")
        
        # Add edges
        self.graph.add_edges_from(edges)
        
        # Add node attributes
        for node in self.graph.nodes():
            full_node_id = "hep-th/"+node
            if full_node_id in self.paper_metadata:
                self.graph.nodes[node].update(self.paper_metadata[full_node_id])
            else:
                self.graph.nodes[node]['paper_id'] = node
                self.graph.nodes[node]['title'] = f"Paper {node}"
                self.graph.nodes[node]['authors'] = ''
                self.graph.nodes[node]['abstract'] = ''
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def export_csv(self, output_dir: str = "data/processed") -> Tuple[Path, Path]:
        """Export nodes and edges as CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export nodes
        nodes_data = []
        for node in self.graph.nodes():
            nodes_data.append({
                'paper_id': node,
                'title': self.graph.nodes[node].get('title', ''),
                # 'authors': self.graph.nodes[node].get('authors', ''),
                # 'abstract': self.graph.nodes[node].get('abstract', '')
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_csv = output_path / "nodes.csv"
        nodes_df.to_csv(nodes_csv, index=False)
        print(f"Exported {len(nodes_df)} nodes to {nodes_csv}")
        
        # Export edges
        edges_data = [{'source': s, 'target': t} for s, t in self.graph.edges()]
        
        edges_df = pd.DataFrame(edges_data)
        edges_csv = output_path / "edges.csv"
        edges_df.to_csv(edges_csv, index=False)
        print(f"Exported {len(edges_df)} edges to {edges_csv}")
        
        return nodes_csv, edges_csv
    
    def export_graphml(self, output_file: str = "data/processed/citation_network.graphml") -> Path:
        """Export graph in GraphML format."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        nx.write_graphml(self.graph, str(output_path))
        print(f"Exported graph to {output_path}")
        
        return output_path


def main():
    """Parse citation network and export to CSV and GraphML formats."""
    parser = CitationNetworkParser()
    
    # Verify files exist
    try:
        parser.verify_files()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Parse data
    edges = parser.parse_citations()
    parser.parse_abstracts()
    
    # Build graph
    parser.build_graph(edges)
    
    # Export formats
    parser.export_csv()
    parser.export_graphml()
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()