from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Output file
output_file = "../outputs/exp06_output.txt"

# -------------------------------
# 1. Knowledge Base
# -------------------------------
documents = [
    "The Eiffel Tower is located in Paris, France and was completed in 1889.",
    "Retrieval-Augmented Generation combines document retrieval with text generation.",
    "Python is a popular high-level programming language used in AI development.",
    "Vector databases store embeddings and support fast similarity search."
]

# -------------------------------
# 2. Generate Document Embeddings
# -------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

# -------------------------------
# 3. Build FAISS Index
# -------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# -------------------------------
# 4. Query and Retrieve Documents
# -------------------------------
query = "What is RAG in AI?"

query_embedding = embed_model.encode([query])

D, I = index.search(np.array(query_embedding), k=2)

retrieved_chunks = [documents[i] for i in I[0]]

# -------------------------------
# 5. Generate Answer
# -------------------------------
context = " ".join(retrieved_chunks)

prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

answer = generator(
    prompt,
    max_length=60,
    do_sample=False
)

generated_answer = answer[0]["generated_text"]

# -------------------------------
# Print Results
# -------------------------------
print("Retrieved Context:")
for chunk in retrieved_chunks:
    print("-", chunk)

print("\nAnswer:")
print(generated_answer)

# -------------------------------
# Save Results
# -------------------------------
with open(output_file, "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 6 - RETRIEVAL AUGMENTED GENERATION (RAG)\n")
    file.write("=" * 70 + "\n\n")

    file.write("Query:\n")
    file.write(query + "\n\n")

    file.write("Retrieved Context:\n")
    for chunk in retrieved_chunks:
        file.write("- " + chunk + "\n")

    file.write("\nGenerated Answer:\n")
    file.write(generated_answer)

print("\nOutput saved successfully to ../outputs/exp06_output.txt")