import sys
from pathlib import Path

# Add project root directory to sys.path to support running as script
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.rag.loader import extract_pdf
from app.rag.chunking import chunk_by_sections
from app.rag.embedding import EmbeddingModel
from app.rag.vector_store import VectorStore


def main():

    pdf_path = r"uploads\my_college_syllabus.pdf"

    # Step 1
    print("Step 1: Extracting PDF...")

    text = extract_pdf(pdf_path)

    with open("extracted_text.txt", "w", encoding="utf-8") as file:
        file.write(text)

    print("[SUCCESS] Extracted text saved")

    # Step 2
    print("\nStep 2: Chunking...")

    chunks = chunk_by_sections(text)

    print(f"[SUCCESS] Total Chunks: {len(chunks)}")

    # Step 3
    print("\nStep 3: Generating Embeddings...")

    embedding_model = EmbeddingModel()

    embeddings = embedding_model.generate_embeddings(chunks)

    print(f"[SUCCESS] Generated {len(embeddings)} embeddings")

    # Step 4
    print("\nStep 4: Storing in ChromaDB...")

    vector_store = VectorStore()

    vector_store.add(
        chunks,
        embeddings,
        source="my_college_syllabus"
    )

    print("[SUCCESS] Vectors stored successfully")


if __name__ == "__main__":
    main()