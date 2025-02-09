#!/usr/bin/env python3
"""
LLM-based PDF Segmentation

This script:
  - Performs OCR on each page of a PDF.
  - Uses an LLM (via Anthropic) to analyze the OCR text and determine which page numbers mark the beginning of new documents.
  - Splits the PDF into separate files based on these boundaries.

Place your PDFs in the folder named 'input_dir' and the outputs will appear under 'output_dir'.
Ensure your Anthropic API key is set in a .env file.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# PDF and OCR libraries
from pdf2image import convert_from_path
import pytesseract

# PDF splitting library
from PyPDF2 import PdfReader, PdfWriter

# Anthropic client (ensure your version is >= 0.4.x)
from anthropic import Anthropic


def convert_page_to_text(pdf_path: str, page_num: int, max_chars: int = 2000) -> str:
    """
    Render a single page (1-indexed) of the PDF as an image,
    run OCR on it, and return up to max_chars characters.
    """
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=150)
        text = pytesseract.image_to_string(images[0])
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        text = ""
    return text[:max_chars]


def segment_document_llm(pdf_path: str, num_pages: int, anth_client, model_name: str = "claude-3-5-sonnet-latest") -> list:
    """
    Uses the LLM to segment the PDF.

    It collects the OCR text for every page, builds a prompt that labels each page with its page number,
    and asks the LLM to return a JSON array of page numbers (1-indexed) that begin new documents.
    
    Returns a list of integers.
    """
    # Extract OCR text from every page.
    page_texts = [convert_page_to_text(pdf_path, i + 1, max_chars=5000) for i in range(num_pages)]
    
    # Build a prompt for the LLM.
    prompt = (
        "You are a document segmentation assistant. Below is the OCR text of a multi-document PDF. "
        "Each page is labeled by its page number. Identify which page numbers mark the beginning of a new document. "
        "Return only a JSON array of integers representing the page numbers (1-indexed) that start new documents. "
        "Lettered or Numbered exhibits are seperate documents"
        "Remember, the first page is always the start of a document.\n\n"
    )
    for i, text in enumerate(page_texts):
        prompt += f"Page {i+1}:\n{text}\n\n"
    prompt += "Please output only a JSON array of page numbers."
    
    print("Sending prompt to LLM for segmentation...")
    try:
        response = anth_client.messages.create(
            model=model_name,
            system="You are a document segmentation assistant.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        answer_block = response.content[0]
        answer = answer_block.text if hasattr(answer_block, "text") else str(answer_block)
        answer = answer.strip()
        print("LLM response:", answer)
        boundaries = json.loads(answer)
        if not isinstance(boundaries, list):
            print("LLM response is not a list.")
            return []
        return boundaries
    except Exception as e:
        print("Error in LLM segmentation:", e)
        return []


def build_segments_from_boundaries(boundaries: list, num_pages: int) -> list:
    """
    Converts the LLM-provided page boundaries (1-indexed) into a list of segments (0-indexed tuples).

    For example, if the LLM returns [1, 5, 9] and there are 12 pages, the segments will be:
      - Segment 1: pages 1 to 4 (0-indexed: (0, 3))
      - Segment 2: pages 5 to 8 (0-indexed: (4, 7))
      - Segment 3: pages 9 to 12 (0-indexed: (8, 11))
    """
    # Ensure the first page is included.
    if 1 not in boundaries:
        boundaries.insert(0, 1)
    boundaries = sorted(boundaries)
    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i] - 1  # convert to 0-indexed
        if i < len(boundaries) - 1:
            end = boundaries[i+1] - 2
        else:
            end = num_pages - 1
        segments.append((start, end))
    return segments


def split_documents(pdf_path: str, segments: list, output_dir: str) -> list:
    """
    Splits the PDF (given by pdf_path) into separate files based on the segments.
    Each segment is defined as a tuple (start_page, end_page) using 0-indexed page numbers.
    
    Returns a list of output file paths.
    """
    output_files = []
    reader = PdfReader(pdf_path)
    pdf_name = Path(pdf_path).stem

    for idx, (start, end) in enumerate(segments):
        writer = PdfWriter()
        for i in range(start, end + 1):
            try:
                writer.add_page(reader.pages[i])
            except IndexError:
                print(f"Page index {i} is out of range for PDF: {pdf_path}")
                continue

        out_path = Path(output_dir) / f"{pdf_name}_segment_{idx+1}.pdf"
        with open(out_path, "wb") as fout:
            writer.write(fout)
        output_files.append(str(out_path))
    
    return output_files


def process_single_pdf(pdf_path: str, output_dir: str, anth_client) -> int:
    """
    Processes a single PDF:
      - Counts the pages.
      - Uses the LLM to determine document boundaries.
      - Converts the boundaries into segments.
      - Splits the PDF into separate files based on these segments.
      
    Returns the number of document segments found.
    """
    pdf_output_dir = Path(output_dir) / Path(pdf_path).stem
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {pdf_path}")
    try:
        images = convert_from_path(pdf_path, dpi=150)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return 0

    num_pages = len(images)
    print(f"Extracted {num_pages} pages from PDF.")

    boundaries = segment_document_llm(pdf_path, num_pages, anth_client)
    print("LLM suggested boundaries (page numbers):", boundaries)
    
    segments = build_segments_from_boundaries(boundaries, num_pages)
    print("Calculated segments (0-indexed):", segments)
    
    output_files = split_documents(pdf_path, segments, str(pdf_output_dir))
    if output_files:
        print("Created the following PDF segment files:")
        for file in output_files:
            print(f"  - {file}")
    else:
        print("No PDF segments were created.")
    
    return len(segments)


def main():
    input_dir = Path("input_dir")
    output_dir = Path("output_dir")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Load environment variables (e.g., Anthropic API key)
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=str(env_path))
    anth_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anth_api_key:
        print("Anthropic API key not found in environment variables.")
        return

    anth_client = Anthropic(api_key=anth_api_key)
    
    print(f"Looking for PDF files in: {input_dir.resolve()}")
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return

    total_segments = 0
    for pdf_file in pdf_files:
        seg_count = process_single_pdf(str(pdf_file), str(output_dir), anth_client)
        total_segments += seg_count

    print("\nProcessing complete.")
    print(f"Processed {len(pdf_files)} PDF file(s) with a total of {total_segments} document segment(s).")
    print("Output directory:", output_dir.resolve())


if __name__ == "__main__":
    main()
