from transformers import pipeline
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Output file
output_file = "../outputs/exp05_output.txt"

# ---------- Text Summarization ----------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Generative AI refers to a class of artificial intelligence models capable of
producing new content such as text, images, audio, and video. Large Language Models (LLMs)
such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of
natural language tasks including translation, summarization, and question answering. These
models are increasingly being deployed in industry applications ranging from customer support
to software development, transforming how humans interact with machines.
"""

summary = summarizer(
    article,
    max_length=45,
    min_length=20,
    do_sample=False
)

# ---------- Question Answering ----------
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

context = article
question = "What are Large Language Models trained on?"

answer = qa(
    question=question,
    context=context
)

# ---------- Print and Save Output ----------
with open("../outputs/exp05_output.txt", "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 5 - TEXT SUMMARIZATION AND QUESTION ANSWERING\n")
    file.write("=" * 65 + "\n\n")

    print("Summary:\n")
    print(summary[0]["summary_text"])

    file.write("Summary:\n")
    file.write(summary[0]["summary_text"] + "\n\n")

    print("\nQuestion:", question)
    print(
        "Answer:",
        answer["answer"],
        "| Confidence:",
        round(answer["score"], 3)
    )

    file.write("Question:\n")
    file.write(question + "\n\n")

    file.write("Answer:\n")
    file.write(answer["answer"] + "\n")

    file.write(
        "Confidence: " +
        str(round(answer["score"], 3))
    )

print("\nOutput saved successfully to ../outputs/exp05_output.txt")