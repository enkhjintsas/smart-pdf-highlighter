import logging
import time

import streamlit as st

from src import generate_highlighted_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the PDF Highlighter tool."""
    st.set_page_config(page_title="Smart PDF Highlighter", page_icon="./photos/icon.png")
    st.title("Smart PDF Highlighter")
    show_description()

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.write("PDF file successfully uploaded.")
        query = st.text_input("Enter your query to highlight relevant sentences:")
        if query:
            process_pdf(uploaded_file, query)
        else:
            st.warning("Please enter a query to proceed.")

def show_description():
    """Display description of functionality and maximum limits."""
    st.write("""Welcome to Smart PDF Highlighter! This tool automatically identifies
        and highlights content within your PDF files that are relevant to your query.
        It utilizes AI techniques, including the OpenAI API, to analyze the text and
        intelligently select key sentences for highlighting based on their relevance to
        your query.""")
    st.write("Maximum Limits: 40 pages, 2000 sentences.")

def process_pdf(uploaded_file, query):
    """Process the uploaded PDF file and generate highlighted PDF."""
    st.write("Generating highlighted PDF based on your query...")
    start_time = time.time()

    with st.spinner("Processing..."):
        result = generate_highlighted_pdf(uploaded_file, query)
        if isinstance(result, str):
            st.error(result)
            logger.error("Error generating highlighted PDF: %s", result)
            return
        else:
            file = result

    end_time = time.time()
    execution_time = end_time - start_time
    st.success(
        f"Highlighted PDF generated successfully in {execution_time:.2f} seconds."
    )

    st.write("Download the highlighted PDF:")
    st.download_button(
        label="Download",
        data=file,
        file_name="highlighted_pdf.pdf",
    )

if __name__ == "__main__":
    main()
