from transformers import pipeline
import os

# Create outputs folder (in the project root)
os.makedirs("../outputs", exist_ok=True)

# Output file
output_file = "../outputs/exp02_output.txt"

with open(output_file, "w", encoding="utf-8") as file:

    # ---------- Sentiment Analysis ----------
    sentiment_analyzer = pipeline("sentiment-analysis")

    reviews = [
        "The new smartphone has an amazing camera and battery life!",
        "The delivery was late and the packaging was damaged."
    ]

    file.write("EXPERIMENT 2 - SENTIMENT ANALYSIS\n")
    file.write("=" * 50 + "\n\n")

    for review in reviews:
        result = sentiment_analyzer(review)[0]

        text = (
            f"Review: {review}\n"
            f"-> {result['label']} ({round(result['score'], 3)})\n\n"
        )

        print(text)
        file.write(text)

    # ---------- Zero-Shot Classification ----------
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    document = "The central bank raised interest rates to control rising inflation."

    candidate_labels = [
        "Politics",
        "Economy",
        "Sports",
        "Technology"
    ]

    classification = classifier(document, candidate_labels)

    file.write("\nEXPERIMENT 2 - ZERO SHOT DOCUMENT CLASSIFICATION\n")
    file.write("=" * 50 + "\n\n")

    print("Document:", document)
    file.write(f"Document: {document}\n")

    for label, score in zip(classification["labels"], classification["scores"]):
        line = f"{label}: {round(score,3)}"
        print(line)
        file.write(line + "\n")

print("\nOutput saved successfully to outputs/exp02_output.txt")