# Similarity Graphs for Citation Datasets

## Overview
This repository focuses on vectorizing citation datasets using various text representation methods to analyze content-based similarity. The datasets used include:

### 1. PMC Open Access
The **PubMed Central (PMC) Open Access** dataset contains full-text biomedical and life sciences articles. It is divided into:
- **Commercial Subset:** Articles available for reuse, including commercial applications.
- **Non-Commercial Subset:** Articles available only for non-commercial purposes.

### 2. S2ORC (Semantic Scholar Open Research Corpus)
The **S2ORC** dataset is a large-scale, publicly available collection of academic papers spanning multiple disciplines. It provides metadata, abstracts, and full-text content for research applications, including citation network analysis.

## Methods Used for Vectorization
To transform textual data into numerical representations, the following methods were employed:

1. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Contextual embeddings for capturing deep semantic relationships.
2. **ColBERT (Contextualized Late Interaction over BERT)**
   - Efficient passage retrieval with late interaction mechanisms.
3. **SPLADE (Sparse Lexical and Expansion Model)**
   - Sparse vector representation using transformer-based expansion.
4. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Traditional statistical method for keyword-based document representation.

## Repository Structure
- `data/` – Processed datasets
- `text_embeddings/` - embeddings results
- `text_vectorisation_models/` – Load models and get embeddings

## Getting Started
To reproduce the results:
1. Clone the repository:
   ```bash
   git clone https://github.com/abhivanth/SimilarityGraphs.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

