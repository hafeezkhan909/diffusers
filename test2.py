import torch
from diffusers import StableDiffusion3Pipeline

# Load pipeline
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

# Load latents
new_prompt = "A white lion on a couch"
saved_data = torch.load("latents_25.pt")
latents_resumed = saved_data["latents"].to("cuda")

# ✅ Resume from latents at step 25
image = pipe(
    new_prompt, 
    num_inference_steps=50, 
    guidance_scale=7.5, 
    latents=latents_resumed,
    resume_step=26,  # ✅ Resume from step 25
    resume_mode=True  # ✅ Enable resuming mode
).images[0]

# Save and display image
image.save("sd3_modified.png")
image.show()

print("Modified image saved as sd3_modified.png")
