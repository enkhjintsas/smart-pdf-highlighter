"""
This module provides functions for generating a highlighted PDF with important sentences.

The main function, `generate_highlighted_pdf`, takes an input PDF file and a pre-trained
sentence embedding model as input.

It splits the text of the PDF into sentences, computes sentence embeddings, and builds a
graph based on the cosine similarity between embeddings. Sentences are clustered, and important
ones are selected based on PageRank scores and clustering.

Finally, the selected sentences are highlighted in the PDF, and the highlighted PDF content
is returned.

Note: This module requires PyMuPDF, networkx, numpy, torch, sentence_transformers, and
sklearn libraries.
"""

import logging
from typing import BinaryIO, List, Tuple

import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import time
import openai
import streamlit as st

# Configure OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
PAGERANK_THRESHOLD_RATIO = 0.15
NUM_CLUSTERS_RATIO = 0.05
MIN_WORDS = 10

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def is_financially_relevant(sentences: List[str]) -> List[bool]:
    """
    Checks if each sentence in a list is financially relevant by batching sentences.
    Args:
        sentences (List[str]): List of sentences to analyze.
    Returns:
        List[bool]: List of booleans indicating relevance for each sentence.
    """
    prompt = (
        "For each sentence below, identify if it is relevant to financial metrics, "
        "market trends, or strategic insights, focusing on financial health indicators "
        "such as cash shortages, leverage reduction, deleveraging, cash runway, debt "
        "repayment, or retirement strategies.\n\n"
    )
    prompt += "\n\n".join([f"Sentence {i+1}: '{sentence}'" for i, sentence in enumerate(sentences)])
    prompt += "\n\nAnswer 'yes' or 'no' for each sentence."

    retry_attempts = 5
    delay = 1

    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for financial analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10 * len(sentences)  # Adjust tokens based on batch size
            )
            # Parse answers for each sentence
            answers = response.choices[0].message['content'].strip().splitlines()
            return ["yes" in answer.lower() for answer in answers]
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    raise Exception("Exceeded maximum retry attempts due to rate limit.")


def load_sentence_model(revision: str = None) -> SentenceTransformer:
    """Load a pre-trained sentence embedding model."""
    return SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)

def encode_sentence(model: SentenceTransformer, sentence: str) -> torch.Tensor:
    """Encode a sentence into a vector representation."""
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        return model.encode(sentence, convert_to_tensor=True).to(device)

def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """Compute the cosine similarity matrix between sentence embeddings."""
    scores = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    )
    similarity_matrix = scores.cpu().numpy()
    normalized_adjacency_matrix = similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True)
    return normalized_adjacency_matrix

def build_graph(normalized_adjacency_matrix: np.ndarray) -> nx.DiGraph:
    """Build a directed graph from a normalized adjacency matrix."""
    return nx.DiGraph(normalized_adjacency_matrix)

def rank_sentences(graph: nx.DiGraph, sentences: List[str]) -> List[Tuple[str, float]]:
    """Rank sentences based on PageRank scores."""
    pagerank_scores = nx.pagerank(graph)
    ranked_sentences = sorted(
        zip(sentences, pagerank_scores.values()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked_sentences

def cluster_sentences(embeddings: torch.Tensor, num_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster sentence embeddings using K-means clustering."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(embeddings.cpu())
    cluster_centers = kmeans.cluster_centers_
    return cluster_assignments, cluster_centers

def get_middle_sentence(cluster_indices: np.ndarray, sentences: List[str]) -> List[str]:
    """Get the middle sentence from each cluster."""
    middle_indices = [
        int(np.median(np.where(cluster_indices == i)[0]))
        for i in range(max(cluster_indices) + 1)
    ]
    middle_sentences = [sentences[i] for i in middle_indices]
    return middle_sentences

def split_text_into_sentences(text: str, min_words: int = MIN_WORDS) -> List[str]:
    """Split text into sentences."""
    sentences = []
    for s in text.split("."):
        s = s.strip()
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / len(s)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc):
    """Generator to yield text per page from the PDF, for memory efficiency."""
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()

def generate_highlighted_pdf(input_pdf_file: BinaryIO, model=load_sentence_model()) -> bytes:
    """Generate a highlighted PDF with important sentences."""
    with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
        num_pages = doc.page_count
        if num_pages > MAX_PAGE:
            return f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages."

        sentences = []
        for page_text in extract_text_from_pages(doc):
            all_sentences = split_text_into_sentences(page_text)
            financial_sentences = [s for s in all_sentences if is_financially_relevant(s)]
            sentences.extend(financial_sentences)

        if len(sentences) > MAX_SENTENCES:
            return f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences."

        embeddings = encode_sentence(model, sentences)
        similarity_matrix = compute_similarity_matrix(embeddings)
        graph = build_graph(similarity_matrix)
        ranked_sentences = rank_sentences(graph, sentences)
        
        pagerank_threshold = int(len(ranked_sentences) * PAGERANK_THRESHOLD_RATIO) + 1
        top_pagerank_sentences = [
            sentence[0] for sentence in ranked_sentences[:pagerank_threshold]
        ]

        for i in range(num_pages):
            try:
                page = doc[i]
                for sentence in top_pagerank_sentences:
                    rects = page.search_for(sentence)
                    for rect in rects:
                        annot = page.add_highlight_annot(rect)
                        annot.update()
            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")

        return doc.write()
