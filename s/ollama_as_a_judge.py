import sys
import chromadb
import ollama

DB_DIR = "./chroma_db"
COLLECTION_NAME = "my_docs"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"             
TOP_K = 4                         

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer isn't contained in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def embed_query(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def retrieve_chunks(query_embedding: list[float], k: int):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_prompt(question: str, retrieved: list) -> str:
    context = "\n\n---\n\n".join(chunk for chunk, _ in retrieved)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(prompt: str) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        return

    question = " ".join(sys.argv[1:])

    print(f"Question: {question}\n")

    print("Embedding query...")
    query_embedding = embed_query(question)

    print(f"Retrieving top {TOP_K} chunks...")
    retrieved = retrieve_chunks(query_embedding, TOP_K)

    print("\nRetrieved sources:")
    for chunk, meta in retrieved:
        preview = chunk[:80].replace("\n", " ")
        print(f"  - {meta['source']} (chunk {meta['chunk_index']}): {preview}...")

    prompt = build_prompt(question, retrieved)

    print("\nGenerating answer...\n")
    answer = generate_answer(prompt)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
