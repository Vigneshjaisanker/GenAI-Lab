from transformers import pipeline, set_seed
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Load GPT-2 model
generator = pipeline("text-generation", model="gpt2")
set_seed(42)

# Input prompt
prompt = "Artificial Intelligence will transform the future of"

# Generate text
outputs = generator(
    prompt,
    max_length=60,
    num_return_sequences=2,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Save output to ../outputs
with open("../outputs/exp01_output.txt", "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 1 - TEXT GENERATION USING GPT-2\n")
    file.write("=" * 50 + "\n\n")
    file.write(f"Prompt:\n{prompt}\n\n")

    for i, out in enumerate(outputs, 1):
        text = out["generated_text"]

        # Print on terminal
        print(f"--- Generated Text {i} ---")
        print(text)
        print()

        # Save in file
        file.write(f"--- Generated Text {i} ---\n")
        file.write(text + "\n\n")

print("Output saved successfully to ../outputs/exp01_output.txt")