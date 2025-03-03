import torch
from diffusers import StableDiffusion3Pipeline
import os
# from qwen_integration import get_refined_prompt

def parse_qwen_output(full_output):
    """Parses Qwen output to extract decision (True/False) and refined prompt."""
    decision = None
    refined_prompt = None

    for line in full_output.split("\n"):
        if line.startswith("DECISION:"):
            decision = line.replace("DECISION:", "").strip().strip('"')
        elif line.startswith("REFINED PROMPT:"):
            refined_prompt = line.replace("REFINED PROMPT:", "").strip().strip('"')
    # Ensure refined_prompt is valid; default to original prompt if missing
    if refined_prompt is None:
        refined_prompt = "None"
    return decision, refined_prompt

# Load Stable Diffusion 3 pipeline
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

### **Step 1: Generate and Save Latents**
initial_prompt = "A white dog on a couch under the sun"
resume_step = 25
refinement_step=37
tag="single_object"

print("\n[Step 1] Generating latents at resume steps and refinement step...")
pipe(
    initial_prompt, 
    num_inference_steps=50, 
    guidance_scale=7.5, 
    resume_step=resume_step,  
    refinement_step=refinement_step,
    save_latents_mode=True  # ✅ Save latents internally
)
print("[Step 1] Latents saved at step 25.\n")

# 
### **Step 2: Resume Generation from Latents**
prev_step_image = os.path.join(".", "Step_75.png")

# full_output = get_refined_prompt(initial_prompt, prev_step_image, tag)
# decision, refined_prompt = parse_qwen_output(full_output)
# print(f"✅ Refining further: Refined Prompt for Step {resume_step}: {refined_prompt}")
decision, refined_prompt = True, "An orange cat on a couch under the sun"

print("[Step 2] Resuming from saved latents at step 25...")
saved_data = torch.load(f"latents_{resume_step}.pt")
latents_resumed = saved_data["latents"].to("cuda")

# ✅ Resume from latents and generate the final image
image = pipe(
    refined_prompt, 
    num_inference_steps=50, 
    guidance_scale=7.5, 
    latents=latents_resumed,
    resume_step=resume_step+1,  # ✅ Start from the next step after saving
    resume_mode=True  # ✅ Resume from latents
).images[0]

# ✅ Save and display the final image
image.save("sd3_modified.png")
image.show()
print("[Step 2] Modified image saved as sd3_modified.png\n")

print("✅ Test complete: Saved latents and resumed successfully!")
