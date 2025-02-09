#!/usr/bin/env python3
"""
Hybrid PDF segmentation:
  - Performs OCR on each page.
  - Computes candidate boundaries using a moving-window semantic similarity approach.
  - For each candidate boundary, asks Anthropic to confirm if it marks a new document.
  - If no candidates are confirmed by Anthropic, falls back to using the raw candidate boundaries.
  - Splits the PDF into separate files based on the confirmed (or fallback) boundaries.
  
Tune the parameters (similarity threshold and window size) to get the desired segmentation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# PDF and OCR libraries
from pdf2image import convert_from_path
import pytesseract

# PDF splitting library
from PyPDF2 import PdfReader, PdfWriter

# For semantic similarity (sentence embeddings)
from sentence_transformers import SentenceTransformer, util
import torch

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


def confirm_boundary(anth_client, prev_text: str, curr_text: str, model_name: str = "claude-3-5-sonnet-latest") -> bool:
    """
    Uses Anthropic to decide if the current page marks a new document.
    The prompt asks whether the current page begins a new document and expects ONLY "YES" or "NO".
    """
    try:
        response = anth_client.messages.create(
            model=model_name,
            system="You are a document segmentation assistant.",
            messages=[{
                "role": "user",
                "content": (
                    "Below are two consecutive pages from a PDF.\n"
                    "Does the second page mark the beginning of a new document? Answer ONLY YES or NO.\n\n"
                    f"Previous page text:\n{prev_text}\n\n"
                    f"Current page text:\n{curr_text}\n"
                ),
            }],
            max_tokens=100,
        )
        answer_block = response.content[0]
        answer = answer_block.text if hasattr(answer_block, "text") else str(answer_block)
        answer = answer.strip().upper()
        return "YES" in answer
    except Exception as e:
        print("Error in Anthropic boundary decision:", e)
        return False


def hybrid_segment_document(pdf_path: str, num_pages: int, anth_client, threshold: float = 0.55, window_size: int = 2) -> (list, list):
    """
    Computes candidate document boundaries by:
      1. Calculating the cosine similarity between the embedding of a moving window
         (the last `window_size` pages of the current segment) and the next page.
      2. If the similarity falls below the threshold, the page is flagged as a candidate boundary.
      3. Each candidate is then confirmed by asking Anthropic.
      4. If no candidates are confirmed by Anthropic, fall back to using the raw candidate boundaries.
      
    Returns:
      - segments: a list of (start_page, end_page) tuples (0-indexed)
      - page_texts: list of OCR text for each page (for debugging or further processing)
    """
    print("Extracting OCR text for each page...")
    page_texts = [convert_page_to_text(pdf_path, i + 1, max_chars=5000) for i in range(num_pages)]
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Computing embeddings for each page...")
    embeddings = model.encode(page_texts, convert_to_tensor=True)
    
    candidate_boundaries = []
    confirmed_boundaries = []
    current_segment_embeddings = [embeddings[0]]
    
    for i in range(1, num_pages):
        # Use the last `window_size` pages from the current segment as the reference.
        if len(current_segment_embeddings) >= window_size:
            window_embeddings = current_segment_embeddings[-window_size:]
        else:
            window_embeddings = current_segment_embeddings
        avg_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
        sim = util.cos_sim(avg_embedding, embeddings[i]).item()
        print(f"Candidate check: similarity for page {i+1}: {sim:.2f}")
        
        if sim < threshold:
            candidate_boundaries.append(i)
            prev_text = page_texts[i - 1]
            curr_text = page_texts[i]
            confirmed = confirm_boundary(anth_client, prev_text, curr_text)
            print(f"Anthropic confirmation for boundary at page {i+1}: {confirmed}")
            if confirmed:
                confirmed_boundaries.append(i)
                current_segment_embeddings = [embeddings[i]]
            else:
                current_segment_embeddings.append(embeddings[i])
        else:
            current_segment_embeddings.append(embeddings[i])
    
    # If Anthropic did not confirm any boundaries, fall back to candidate boundaries.
    if not confirmed_boundaries and candidate_boundaries:
        print("No boundaries confirmed by Anthropic; falling back to candidate boundaries.")
        confirmed_boundaries = candidate_boundaries

    # Build segments from confirmed boundaries.
    segments = []
    start = 0
    for boundary in confirmed_boundaries:
        segments.append((start, boundary - 1))
        start = boundary
    segments.append((start, num_pages - 1))
    
    return segments, page_texts


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


def process_single_pdf(pdf_path: str, output_dir: str, anth_client, threshold: float = 0.55, window_size: int = 2) -> int:
    """
    Processes a single PDF:
      - Counts the pages.
      - Uses the hybrid segmentation approach (semantic candidate boundaries confirmed by Anthropic)
        to determine document segments.
      - Splits the PDF into separate files based on the detected segments.
      
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

    print("Segmenting PDF into document segments using hybrid method...")
    segments, _ = hybrid_segment_document(pdf_path, num_pages, anth_client, threshold, window_size)
    print("Detected segments (1-indexed page numbers):")
    for seg in segments:
        print(f"  - Pages {seg[0] + 1} to {seg[1] + 1}")
    
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
    # Adjust threshold and window_size as needed.
    similarity_threshold = 0.55
    window_size = 2

    for pdf_file in pdf_files:
        try:
            seg_count = process_single_pdf(str(pdf_file), str(output_dir), anth_client,
                                           threshold=similarity_threshold, window_size=window_size)
            total_segments += seg_count
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print("\nProcessing complete.")
    print(f"Processed {len(pdf_files)} PDF file(s) with a total of {total_segments} document segment(s).")
    print("Output directory:", output_dir.resolve())


if __name__ == "__main__":
    main()
