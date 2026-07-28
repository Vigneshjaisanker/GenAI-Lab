from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Output file
output_file = "../outputs/exp07_output.txt"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")


def generate_code(prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# 1. Code Generation
# -----------------------------
prompt1 = """# Write a Python function to check if a number is prime
def is_prime(n):"""

generated_function = generate_code(prompt1)

# -----------------------------
# 2. Debugging Example
# -----------------------------
buggy_code = """# The following function should return the factorial of n, but has a bug. Fix it.
def factorial(n):
    result = 0
    for i in range(1, n + 1):
        result = result * i
    return result

# Corrected function:
def factorial_fixed(n):"""

debug_suggestion = generate_code(
    buggy_code,
    max_new_tokens=60
)

# -----------------------------
# Print Output
# -----------------------------
print("Generated Function:\n")
print(generated_function)

print("\n" + "=" * 70 + "\n")

print("Debug Suggestion:\n")
print(debug_suggestion)

# -----------------------------
# Save Output
# -----------------------------
with open(output_file, "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 7 - AI CODE GENERATION AND DEBUGGING\n")
    file.write("=" * 70 + "\n\n")

    file.write("Generated Function:\n\n")
    file.write(generated_function)

    file.write("\n\n")
    file.write("=" * 70)
    file.write("\n\n")

    file.write("Debug Suggestion:\n\n")
    file.write(debug_suggestion)

print("\nOutput saved successfully to ../outputs/exp07_output.txt")