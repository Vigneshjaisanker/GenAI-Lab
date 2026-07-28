from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Output file
output_file = "../outputs/exp04_output.txt"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

print("Chatbot ready! Type 'quit' to exit.")

with open(output_file, "w", encoding="utf-8") as file:

    file.write("EXPERIMENT 4 - DIALOGGPT CHATBOT\n")
    file.write("=" * 50 + "\n\n")

    for step in range(5):

        user_input = input(">> User: ")

        if user_input.lower() == "quit":
            break

        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors="pt"
        )

        if chat_history_ids is not None:
            bot_input_ids = torch.cat(
                [chat_history_ids, new_input_ids],
                dim=-1
            )
        else:
            bot_input_ids = new_input_ids

        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9
        )

        response = tokenizer.decode(
            chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        print("Bot:", response)

        # Save conversation
        file.write(f"User: {user_input}\n")
        file.write(f"Bot: {response}\n\n")

print("\nConversation saved successfully to ../outputs/exp04_output.txt")