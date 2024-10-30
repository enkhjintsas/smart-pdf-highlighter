"""
This module provides functions for generating a highlighted PDF with important sentences.

The main function, `generate_highlighted_pdf`, takes an input PDF file and generates a
highlighted version of the PDF based on the financial relevance of sentences.

It splits the text of the PDF into sentences, determines financial relevance using OpenAI's API,
and highlights the most important sentences.

Note: This module requires PyMuPDF, networkx, numpy, torch, sentence_transformers, openai,
and sklearn libraries.
"""

import logging
from typing import BinaryIO, List, Tuple
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import time
import openai
import tiktoken  # For token counting
import random
import streamlit as st  # Import Streamlit to access secrets
from huggingface_hub import hf_hub_download  # Updated import

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
MIN_WORDS = 10
PAGERANK_THRESHOLD_RATIO = 0.15
MODEL_NAME = "gpt-3.5-turbo"
MAX_MODEL_TOKENS = 4096  # Max tokens for gpt-3.5-turbo
MAX_PROMPT_TOKENS = 3500  # Leave room for response tokens

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Retrieve the OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def count_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """
    Counts the number of tokens in the given text for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def chunk_sentences(sentences: List[str], max_tokens: int) -> List[List[str]]:
    """
    Splits a list of sentences into chunks that stay within the token limit.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        # Add tokens for the prompt formatting per sentence
        prompt_tokens = count_tokens(f"Sentence: {sentence}\nAnswer:") + 2  # Extra tokens for formatting
        total_tokens = sentence_tokens + prompt_tokens

        if current_tokens + total_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [sentence]
            current_tokens = total_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += total_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def is_financially_relevant_batch(sentences: List[str]) -> List[bool]:
    """
    Determines if each sentence in the batch is financially relevant.

    Args:
        sentences (List[str]): List of sentences to analyze.

    Returns:
        List[bool]: List indicating relevance for each sentence.
    """
    prompt_header = (
        "Identify whether each of the following sentences is relevant to financial metrics, "
        "market trends, or strategic insights, focusing on financial health indicators like "
        "cash shortages, leverage reduction, deleveraging, cash runway, debt repayment, or "
        "retirement strategies.\n\n"
    )

    # Prepare sentences for the prompt
    prompt_body = ""
    for i, sentence in enumerate(sentences):
        prompt_body += f"Sentence {i+1}: {sentence}\nAnswer (yes or no):\n"

    full_prompt = prompt_header + prompt_body

    # Estimate total tokens
    prompt_tokens = count_tokens(full_prompt)
    max_response_tokens = 2 * len(sentences)  # Estimate 2 tokens per response
    total_tokens = prompt_tokens + max_response_tokens

    if total_tokens > MAX_MODEL_TOKENS:
        raise ValueError("The combined prompt and expected response exceed the token limit.")

    # Retry logic with exponential backoff
    retry_attempts = 5
    delay = 5  # Start with a 5-second delay

    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an assistant for financial analysis."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_response_tokens,
                temperature=0.0,
            )
            # Parse the responses
            answers = response.choices[0].message['content'].strip().split('\n')
            results = []
            for answer in answers:
                if "yes" in answer.lower():
                    results.append(True)
                else:
                    results.append(False)
            return results
        except (openai.error.OpenAIError, Exception) as e:
            jitter = random.uniform(0, 3)
            print(f"API error: {e}. Retrying in {delay + jitter:.2f} seconds...")
            time.sleep(delay + jitter)
            delay *= 2  # Exponential backoff

    raise Exception("Exceeded maximum retry attempts due to API errors.")

def load_sentence_model() -> SentenceTransformer:
    """
    Load a pre-trained sentence embedding model.
    """
    # Use hf_hub_download to download the model files
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_path = hf_hub_download(repo_id=model_name, filename='pytorch_model.bin')
    config_path = hf_hub_download(repo_id=model_name, filename='config.json')
    tokenizer_config_path = hf_hub_download(repo_id=model_name, filename='tokenizer_config.json')
    vocab_path = hf_hub_download(repo_id=model_name, filename='vocab.txt')

    # Initialize the model using the downloaded files
    model = SentenceTransformer(model_name_or_path=model_name, cache_folder=None)
    return model

def encode_sentences(model: SentenceTransformer, sentences: List[str]) -> torch.Tensor:
    """
    Encode a list of sentences into embeddings.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False).to(device)
    return embeddings

def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity matrix between embeddings.
    """
    similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
    ).cpu().numpy()
    # Normalize the matrix
    normalized_adjacency_matrix = similarity_matrix / np.sum(similarity_matrix, axis=1, keepdims=True)
    return normalized_adjacency_matrix

def build_graph(normalized_adjacency_matrix: np.ndarray) -> nx.DiGraph:
    """
    Build a graph from the normalized adjacency matrix.
    """
    return nx.from_numpy_array(normalized_adjacency_matrix, create_using=nx.DiGraph)

def rank_sentences(graph: nx.DiGraph, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences using the PageRank algorithm.
    """
    pagerank_scores = nx.pagerank(graph)
    ranked_sentences = sorted(
        ((sentences[i], score) for i, score in pagerank_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked_sentences

def split_text_into_sentences(text: str, min_words: int = MIN_WORDS) -> List[str]:
    """
    Split text into sentences, filtering out short ones.
    """
    import re
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    raw_sentences = sentence_endings.split(text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / max(len(s), 1)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc) -> List[str]:
    """
    Extract text from each page of the PDF.
    """
    page_texts = []
    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text()
        page_texts.append(page_text)
    return page_texts

def generate_highlighted_pdf(input_pdf_file: BinaryIO) -> bytes:
    """
    Generate a PDF with important sentences highlighted.

    Args:
        input_pdf_file (BinaryIO): Input PDF file object.

    Returns:
        bytes: The content of the highlighted PDF.
    """
    model = load_sentence_model()

    with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
        num_pages = doc.page_count

        if num_pages > MAX_PAGE:
            raise ValueError(f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages.")

        # Extract text from all pages
        page_texts = extract_text_from_pages(doc)

        # Split and collect all sentences
        all_sentences = []
        for page_text in page_texts:
            sentences = split_text_into_sentences(page_text)
            all_sentences.extend(sentences)

        if len(all_sentences) > MAX_SENTENCES:
            raise ValueError(f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences.")

        # Chunk sentences to handle token limits
        sentence_chunks = chunk_sentences(all_sentences, MAX_PROMPT_TOKENS)

        # Identify financially relevant sentences
        financial_sentences = []
        for chunk in sentence_chunks:
            relevance = is_financially_relevant_batch(chunk)
            relevant_sentences = [s for s, r in zip(chunk, relevance) if r]
            financial_sentences.extend(relevant_sentences)

        if not financial_sentences:
            raise ValueError("No financially relevant sentences found in the document.")

        # Encode financially relevant sentences
        embeddings = encode_sentences(model, financial_sentences)
        similarity_matrix = compute_similarity_matrix(embeddings)
        graph = build_graph(similarity_matrix)
        ranked_sentences = rank_sentences(graph, financial_sentences)

        # Select top-ranked sentences
        top_n = max(1, int(len(ranked_sentences) * PAGERANK_THRESHOLD_RATIO))
        top_sentences = [s for s, _ in ranked_sentences[:top_n]]

        # Highlight sentences in the PDF
        for i in range(num_pages):
            page = doc[i]
            page_text = page.get_text()
            for sentence in top_sentences:
                if sentence in page_text:
                    rects = page.search_for(sentence)
                    for rect in rects:
                        annot = page.add_highlight_annot(rect)
                        annot.update()

        return doc.write()
