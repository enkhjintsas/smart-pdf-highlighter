import logging
from typing import BinaryIO, List

import fitz  # PyMuPDF
import numpy as np
import openai
import streamlit as st  # Import Streamlit to access secrets

# Constants
MAX_PAGE = 40
MAX_SENTENCES = 2000
PAGERANK_THRESHOLD_RATIO = 0.15
MIN_WORDS = 10

# Logger configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Set up OpenAI API key using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def compute_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Compute embeddings for the sentences using OpenAI API.

    Args:
        sentences (List[str]): List of sentences.

    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = []
    batch_size = 100  # Adjust as appropriate
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-ada-002"
        )
        batch_embeddings = [data['embedding'] for data in response['data']]
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    return embeddings

def split_text_into_sentences(text: str, min_words: int = MIN_WORDS) -> List[str]:
    """
    Split text into sentences.

    Args:
        text (str): Input text.
        min_words (int): Minimum number of words for a valid sentence.

    Returns:
        List[str]: List of sentences.
    """
    sentences = []
    for s in text.split("."):
        s = s.strip()
        # Filtering out short sentences and sentences that contain more than 40% digits
        if (
            s
            and len(s.split()) >= min_words
            and (sum(c.isdigit() for c in s) / len(s)) < 0.4
        ):
            sentences.append(s)
    return sentences

def extract_text_from_pages(doc):
    """Generator to yield text per page from the PDF, for memory efficiency for large PDFs."""
    for page_num in range(len(doc)):
        yield doc[page_num].get_text()

def generate_highlighted_pdf(
    input_pdf_file: BinaryIO, query: str
) -> bytes:
    """
    Generate a highlighted PDF with sentences relevant to the query.

    Args:
        input_pdf_file: Input PDF file object.
        query (str): The query to compute relevance.

    Returns:
        bytes: Highlighted PDF content.
    """
    with fitz.open(stream=input_pdf_file.read(), filetype="pdf") as doc:
        num_pages = doc.page_count

        if num_pages > MAX_PAGE:
            return f"The PDF file exceeds the maximum limit of {MAX_PAGE} pages."

        sentences = []
        for page_text in extract_text_from_pages(doc):
            sentences.extend(split_text_into_sentences(page_text))

        len_sentences = len(sentences)

        print(len_sentences)

        if len_sentences > MAX_SENTENCES:
            return (
                f"The PDF file exceeds the maximum limit of {MAX_SENTENCES} sentences."
            )

        # Compute embeddings for the sentences
        embeddings = compute_embeddings(sentences)
        # Compute embedding for the query
        query_embedding = compute_embeddings([query])[0]

        # Compute cosine similarity between query and each sentence embedding
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
        similarity_scores = np.dot(embeddings_norm, query_embedding_norm)

        # Rank sentences based on similarity scores
        ranked_indices = np.argsort(-similarity_scores)
        ranked_sentences = [sentences[i] for i in ranked_indices]
        ranked_scores = [similarity_scores[i] for i in ranked_indices]

        # Select top sentences based on a threshold or top N
        top_threshold = int(len(ranked_sentences) * PAGERANK_THRESHOLD_RATIO) + 1
        top_sentences = ranked_sentences[:top_threshold]

        # Highlight the top sentences in the PDF
        for i in range(num_pages):
            try:
                page = doc[i]

                for sentence in top_sentences:
                    rects = page.search_for(sentence)
                    colors = (fitz.pdfcolor["yellow"], fitz.pdfcolor["green"])

                    for j, rect in enumerate(rects):
                        color = colors[j % 2]
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=color)
                        annot.update()
            except Exception as e:
                logger.error(f"Error processing page {i}: {e}")

        output_pdf = doc.write()

    return output_pdf
