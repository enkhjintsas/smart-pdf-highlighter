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
import random
import tiktoken  # For token counting

# Configure OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
MIN_WORDS = 10
PAGERANK_THRESHOLD_RATIO = 0.15
NUM_CLUSTERS_RATIO = 0.05
MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 4096  # Max tokens for gpt-3.5-turbo
CHUNK_SIZE = 2000  # Adjust based on your needs and model limits

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def count_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """
    Counts the number of tokens in the given text for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def chunk_sentences(sentences: List[str], max_chunk_size: int) -> List[List[str]]:
    """
    Splits a list of sentences into chunks that stay within the token limit.
    """
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

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
        "For each sentence below, identify if it is relevant to financial metrics, "
        "market trends, or strategic insights, focusing on financial health indicators "
        "such as cash shortages, leverage reduction, deleveraging, cash runway, debt "
        "repayment, or retirement strategies.\n\n"
    )

    # Prepare sentences for the prompt
    sentence_list = [f"Sentence {i+1}: {sentence}" for i, sentence in enumerate(sentences)]
    prompt_body = "\n".join(sentence_list)
    prompt_footer = "\n\nRespond with 'yes' or 'no' for each sentence in order, one per line."

    full_prompt = prompt_header + prompt_body + prompt_footer

    # Estimate total tokens
    prompt_tokens = count_tokens(full_prompt)
    max_response_tokens = 2 * len(sentences)  # Estimate 2 tokens per response
    total_tokens = prompt_tokens + max_response_tokens

    if total_tokens > MAX_TOKENS:
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
            answers = response.choices[0].message['content'].strip().splitlines()
            return ["yes" in answer.lower() for answer in answers]
        except (openai.error.OpenAIError, Exception) as e:
            jitter = random.uniform(0, 3)
            print(f"API error: {e}. Retrying in {delay + jitter:.2f} seconds...")
            time.sleep(delay + jitter)
            delay *= 2  # Exponential backoff

    raise Exception("Exceeded maximum retry attempts due to API errors.")

def load_sentence_model(revision: str = None) -> SentenceTransformer:
    """
    Load a pre-trained sentence embedding model.
    """
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision=revision)

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
    similarity_matrix = torch.matmul(embeddings, embeddings.T).cpu().numpy()
    normalized_adjacency_matrix = similarity_matrix / np.sum(similarity_matrix, axis=1, keepdims=True)
    return normalized_adjacency_matrix

def build_graph(normalized_adjacency_matrix: np.ndarray) -> nx.DiGraph:
    """
    Build a graph from the normalized adjacency matrix.
    """
    return nx.from_numpy_array(normalized_adjacency_matrix, create_using=nx.DiGraph)

def rank_sentences(graph: nx.DiGraph, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences using PageRank algorithm.
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
    sentences = []
    for s in text.replace('\n', ' ').split('.'):
        s = s.strip()
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / max(len(s), 1)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc):
    """
    Generator to extract text from each page of the PDF.
    """
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()

def generate_highlighted_pdf(input_pdf_file: BinaryIO, model=None) -> bytes:
    """
    Generate a PDF with important sentences highlighted.
    """
    if model is None:
        model = load_sentence_model()

    with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
        num_pages = doc.page_count

        if num_pages > MAX_PAGE:
            return f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages."

        all_sentences = []
        for page_text in extract_text_from_pages(doc):
            sentences = split_text_into_sentences(page_text)
            all_sentences.extend(sentences)

        if len(all_sentences) > MAX_SENTENCES:
            return f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences."

        # Chunk sentences to handle token limits
        sentence_chunks = chunk_sentences(all_sentences, CHUNK_SIZE)

        # Identify financially relevant sentences
        financial_sentences = []
        for chunk in sentence_chunks:
            relevance = is_financially_relevant_batch(chunk)
            financial_sentences.extend([s for s, r in zip(chunk, relevance) if r])

        if not financial_sentences:
            return "No financially relevant sentences found in the document."

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
