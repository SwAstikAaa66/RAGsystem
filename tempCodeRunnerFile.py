
API_URL = (
    "https://api-inference.huggingface.co/models/"
    "microsoft/Phi-3-mini-4k-instruct"
)

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------------------
# Embedding Model
# ---------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ---------------------------
# Read Documents
# ---------------------------

with open(
    "Data/sample.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

# ---------------------------
# Chunking
# ---------------------------

chunk_size = 500

chunks = [
    text[i:i+chunk_size]
    for i in range(0, len(text), chunk_size)
]

print(f"Created {len(chunks)} chunks")

# ---------------------------
# Embeddings
# ---------------------------

embeddings = embedding_model.encode(chunks)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

print("Knowledge Base Ready!")

# ---------------------------
# Ask Questions
# ---------------------------

while True:

    query = input(
        "\nAsk a question (or type exit): "
    )

    if query.lower() == "exit":
        break

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        k=2
    )

    context = "\n".join(
        [chunks[i] for i in indices[0]]
    )

    prompt = f"""
Use only the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    result = response.json()