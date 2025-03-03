import torch
import os
from diffusers import StableDiffusion3Pipeline

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

# Load pipeline
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

# ✅ Step 1: Generate and Save Latents for Multiple Resume Steps
initial_prompt = "A white dog on a couch under the sun"
restart_steps = [20, 10, 0]  # ✅ Now a list of resume steps
refinement_step = 40
tag = "single_object"

print("\n[Step 1] Generating latents at resume steps and refinement step...")
image = pipe(
    initial_prompt, 
    num_inference_steps=100, 
    guidance_scale=7.5, 
    restart_steps=restart_steps,  # ✅ Pass list of restart steps
    vanilla_generation=True,
    refinement_step=refinement_step,
    save_latents_mode=True  # ✅ Save latents internally
).images[0]
print("[Step 1] Latents saved at specified steps.\n")

initial_image_path = f"test/final_image.png"
image.save(initial_image_path)
print(f"Step 1 image saved as {initial_image_path}\n")

# ✅ Step 2: Iteratively Resume Generation for Each Resume Step
current_prompt = initial_prompt  # Start with original prompt
prev_refinement_image = f"test/initial_step_{refinement_step}.png"  # First iteration uses Step_37 from initial generation

for idx, step in enumerate(restart_steps):
    print(f"\n[Step {idx + 2}] Using refinement image: {prev_refinement_image} for Qwen")

    # ✅ Uncomment when Qwen is integrated
    # full_output = get_refined_prompt(current_prompt, prev_refinement_image, tag)
    # decision, refined_prompt = parse_qwen_output(full_output)
    
    # ✅ For now, just use a random refined prompt
    if idx == 0:
        decision, refined_prompt = True, f"A white cat on a couch under the sun"
    elif idx == 1:
        decision, refined_prompt = True, f"A white bird on a couch under the sun"
    else:
        decision, refined_prompt = True, f"A white lion on a couch under the sun"

    print(f"[Step {idx + 2}] Refining from saved latents at step {step} with new prompt: {refined_prompt}...")
    saved_data = torch.load(f"test/latents_{step}.pt")
    latents_resumed = saved_data["latents"].to("cuda")

    # ✅ Resume from latents and generate the final image
    image = pipe(
        refined_prompt, 
        num_inference_steps=100, 
        guidance_scale=7.5, 
        latents=latents_resumed,
        resume_step=step + 1,  # ✅ Start from the next step of the saved latent
        refinement_step=refinement_step,  # ✅ Save new refinement image
        resume_mode=True  # ✅ Resume from latents
    ).images[0]

    # ✅ Save the modified image final_{idx+1}_refined_{step}_
    refined_image_path = f"test/final_{idx+1}_refined_{step}_.png"
    image.save(refined_image_path)
    print(f"[Step {idx + 2}] Refined image saved as {refined_image_path}\n")

    # ✅ Update the refinement image path for the next iteration
    prev_refinement_image = refined_image_path

print("✅ Full test complete: Saved latents, refined prompts, and resumed successfully!")
