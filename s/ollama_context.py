import os
import glob
import chromadb
import ollama

DOCS_DIR = "./docs"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
EMBED_MODEL = "nomic-embed-text"   # run: ollama pull nomic-embed-text

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(docs_dir: str) -> list[dict]:
    """Read all .txt and .md files, return list of {text, source}."""
    documents = []
    paths = glob.glob(os.path.join(docs_dir, "**", "*.txt"), recursive=True)
    paths += glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"text": text, "source": os.path.basename(path)})

    return documents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_text(text: str) -> list[float]:
    """Get an embedding vector for a piece of text via Ollama."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def main():
    print(f"Loading documents from {DOCS_DIR}...")
    documents = load_documents(DOCS_DIR)
    if not documents:
        print(f"No .txt or .md files found in {DOCS_DIR}. Add some and re-run.")
        return
    print(f"Loaded {len(documents)} document(s).")

    # Set up (or reset) the Chroma collection
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, texts, metadatas, embeddings = [], [], [], []
    chunk_counter = 0

    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  {doc['source']}: {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}_{i}"
            embedding = embed_text(chunk)

            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})
            embeddings.append(embedding)
            chunk_counter += 1

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"\nDone. Stored {chunk_counter} chunks in '{DB_DIR}'.")


if __name__ == "__main__":
    main()
