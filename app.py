"""
Smart PDF Highlighter
This script provides a Streamlit web application for automatically identifying and
highlighting important content within PDF files. It utilizes AI techniques such as
deep learning, clustering, and advanced algorithms such as PageRank to analyze text
and intelligently select key sentences for highlighting.

Author: Farzad Salajegheh
Date: 2024
"""

import logging
import time
import tempfile

import streamlit as st
import base64
import streamlit.components.v1 as components


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
        process_pdf(uploaded_file)


def show_description():
    """Display description of functionality and maximum limits."""
    st.write("""Welcome to Smart PDF Highlighter! This tool automatically identifies
        and highlights important content within your PDF files. It utilizes many
        AI techniques such as deep learning and other advanced algorithms to 
        analyze the text and intelligently select key sentences for highlighting.""")
    st.write("Maximum Limits: 40 pages, 2000 sentences.")


# def process_pdf(uploaded_file):
#     """Process the uploaded PDF file and generate highlighted PDF."""
#     st.write("Generating highlighted PDF...")
#     start_time = time.time()

#     with st.spinner("Processing..."):
#         result = generate_highlighted_pdf(uploaded_file)
#         if isinstance(result, str):
#             st.error(result)
#             logger.error("Error generating highlighted PDF: %s", result)
#             return
#         else:
#             file = result

#     end_time = time.time()
#     execution_time = end_time - start_time
#     st.success(
#         f"Highlighted PDF generated successfully in {execution_time:.2f} seconds."
#     )

#     st.write("Download the highlighted PDF:")
#     st.download_button(
#         label="Download",
#         data=file,
#         file_name="highlighted_pdf.pdf",
#     )



def pdf_viewer(pdf_path):
    """Display PDF in an HTML component viewer."""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    components.html(pdf_display, height=900)
    
def process_pdf(uploaded_file):
    """Process the uploaded PDF file and generate highlighted PDF."""
    st.write("Generating highlighted PDF...")
    start_time = time.time()

    with st.spinner("Processing..."):
        result = generate_highlighted_pdf(uploaded_file)
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

    # Save the highlighted PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file)
        temp_file_path = temp_file.name

    # Embed PDF Viewer using iframe
    st.write("Preview the highlighted PDF:")
    pdf_viewer(temp_file_path)

    # Provide download option
    st.write("Download the highlighted PDF:")
    st.download_button(
        label="Download",
        data=file,
        file_name="highlighted_pdf.pdf",
    )



if __name__ == "__main__":
    main()
