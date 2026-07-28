from transformers import pipeline
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")

# Output file
output_file = "../outputs/exp03_output.txt"

# 1. Zero-shot prompt
zero_shot_prompt = (
    "Classify the sentiment of this review as Positive or Negative: "
    "'The product quality is excellent!'\nSentiment:"
)

# 2. Few-shot prompt
few_shot_prompt = """Review: 'I loved this movie, it was fantastic.'
Sentiment: Positive

Review: 'The service was slow and disappointing.'
Sentiment: Negative

Review: 'The product quality is excellent!'
Sentiment:"""

# 3. Chain-of-Thought prompt
cot_prompt = """Q: A shop had 15 apples. It sold 6 and then received 10 more. How many apples now?
A: Let's think step by step. 15 - 6 = 9. 9 + 10 = 19. The answer is 19.

Q: A library had 120 books. It lent out 45 and bought 30 new books. How many books now?
A: Let's think step by step."""

# Save output
with open(output_file, "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 3 - PROMPT ENGINEERING\n")
    file.write("=" * 60 + "\n\n")

    prompts = [
        ("Zero-shot", zero_shot_prompt),
        ("Few-shot", few_shot_prompt),
        ("Chain-of-Thought", cot_prompt)
    ]

    for name, prompt in prompts:

        output = generator(
            prompt,
            max_length=len(prompt.split()) + 40,
            num_return_sequences=1,
            do_sample=False
        )

        generated_text = output[0]["generated_text"]

        # Print on terminal
        print(f"=== {name} ===")
        print(generated_text)
        print()

        # Save to file
        file.write(f"=== {name} ===\n")
        file.write(generated_text)
        file.write("\n\n")

print("Output saved successfully to ../outputs/exp03_output.txt")