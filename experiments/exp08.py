from diffusers import StableDiffusionPipeline
import torch
import os

# Create outputs folder outside experiments
os.makedirs("../outputs", exist_ok=True)

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Load model
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

pipe = pipe.to(device)

# Prompt
prompt = "A futuristic city skyline at sunset, digital art, highly detailed"

# Generate image
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# Save image
output_path = "../outputs/generated_city.png"
image.save(output_path)

print(f"Image generated and saved as {output_path}")