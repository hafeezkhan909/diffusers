import torch
import os
from diffusers import StableDiffusion3Pipeline
import argparse
import json
from qwen_integration import get_refined_prompt
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Run the diffusion pipeline with user-defined restart points.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="filtered_prompts.json",
        help="Path to the prompts file (default: filtered_prompts.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--restart_steps",
        type=int,
        nargs="+",
        default=[0, 0, 0],
        help="List of restart steps (e.g., --restart_steps 25 10 0)"
    )
    parser.add_argument(
        "--refinement_step",
        type=int,
        default=75,
        help="Step at which to take feedback from Qwen (default: 75)"
    )
    return parser.parse_args()

def load_prompts(file_path):
    """Loads prompts from JSON file grouped by tag."""
    with open(file_path, "r") as f:
        return json.load(f)
    
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

def move_files(file_map):
    """Moves files from source to destination paths."""
    for src, dest in file_map.items():
        if os.path.exists(src):
            os.rename(src, dest)

# Load pipeline
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU

# ======================== #
# 🔹 Tag-Specific Logic
# ======================== #
def process_tag(tag, prompts, output_dir, restart_steps, refinement_step):
    """Process all prompts for a specific tag."""
    print(f"\n🔹 Processing tag: {tag}")
    tag_output_dir = os.path.join(output_dir, tag)
    os.makedirs(tag_output_dir, exist_ok=True)

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        line_number = prompt_data["line_number"]  # Get the line number
        print(f"\n🚀 Processing Prompt {line_number} for tag {tag}:\n{prompt}")

        # Create unique prompt-specific folder using the line number
        prompt_id = f"prompt_{line_number:03d}"  # Use line number for folder name
        prompt_output_dir = os.path.join(tag_output_dir, prompt_id)
        os.makedirs(prompt_output_dir, exist_ok=True)

        
        # ✅ Step 1: Generate and Save Latents for Multiple Resume Steps
        # initial_prompt = "A white dog on a couch under the sun"
        index = list(range(len(restart_steps)))  # ✅ Generate index dynamically
        print("\n[Step 1] Generating latents at resume steps and refinement step...")
        image = pipe(
            prompt, 
            num_inference_steps=100, 
            guidance_scale=7.5, 
            restart_steps=restart_steps,  # ✅ Pass list of restart steps
            vanilla_generation=True,
            refinement_step=refinement_step,
            output_dir=prompt_output_dir,
            save_latents_mode=True , # ✅ Save latents internally
            save_latents_refinement_mode=True
        ).images[0]
        print("[Step 1] Latents saved at specified steps.\n")

        initial_image_path = os.path.join(prompt_output_dir, "final_image.png")
        image.save(initial_image_path)
        print(f"Step 1 image saved as {initial_image_path}\n")

        # ✅ Step 2: Iteratively Resume Generation for Each Resume Step
        prev_refinement_image = os.path.join(prompt_output_dir, f"initial_step_{refinement_step}.png")  # First iteration uses Step_37 from initial generation
        
        for idx, step in enumerate(restart_steps):
            print(f"\n[Step {idx + 2}] Using refinement image: {prev_refinement_image} for Qwen")

            # ✅ Uncomment when Qwen is integrated
            full_output = get_refined_prompt(prompt, prev_refinement_image, tag)
            decision, refined_prompt = parse_qwen_output(full_output)
            print(f"✅ Refining further: Refined Prompt for Step {step}: {refined_prompt}")

            # Save Qwen output
            refined_prompt_file = os.path.join(prompt_output_dir, f"{idx}_refined_prompt_{step}_.txt")
            with open(refined_prompt_file, "w") as f:
                f.write(full_output)

            if idx == 0:
              if decision == "True":
                    print(f"✅ Early stopping at {idx+1} iter, image matches the prompt.")
                    shutil.copy(os.path.join(prompt_output_dir, "final_image.png"), os.path.join(prompt_output_dir, f"ES{idx+1}_final_image.png"))
            else:
                if decision == "True":
                    print(f"✅ Early stopping at {idx+1} iter, image matches the prompt.")
                    prev_restart_step = restart_steps[idx - 1]  # Get the previous restart step
                    shutil.copy(os.path.join(prompt_output_dir, f"final_{idx}_refined_{prev_restart_step}_.png"), os.path.join(prompt_output_dir, f"ES{idx+1}_final_image.png"))


            # ✅ For now, just use a random refined prompt
            # if idx == 0:
            #     decision, refined_prompt = True, f"A white cat on a couch under the sun"
            # elif idx == 1:
            #     decision, refined_prompt = True, f"A white bird on a couch under the sun"
            # else:
            #     decision, refined_prompt = True, f"A white lion on a couch under the sun"

            print(f"[Step {idx + 2}] Refining from saved latents at step {step} with new prompt: {refined_prompt}...")
            saved_data = torch.load(os.path.join(prompt_output_dir, f"latents_{step}.pt"))
            latents_resumed = saved_data["latents"].to("cuda")

            # ✅ Resume from latents and generate the final image
            image = pipe(
                refined_prompt, 
                num_inference_steps=100, 
                guidance_scale=7.5, 
                latents=latents_resumed,
                index=index,
                restart_steps=restart_steps,
                resume_step=step + 1,  # ✅ Start from the next step of the saved latent
                refinement_step=refinement_step,  # ✅ Save new refinement image
                resume_mode=True  # ✅ Resume from latents
            ).images[0]

            # ✅ Save the modified image final_{idx+1}_refined_{step}_
            refined_image_path = os.path.join(prompt_output_dir, f"final_{idx+1}_refined_{step}_.png")
            image.save(refined_image_path)
            print(f"[Step {idx + 2}] Refined image saved as {refined_image_path}\n")

            prev_latent_image_path = os.path.join(prompt_output_dir, f"{step}_refined_{refinement_step}_.png")
            # ✅ Update the refinement image path for the next iteration
            prev_refinement_image = prev_latent_image_path

        print(f"\n✅ Completed processing for tag: {tag}")

def main():
    args = parse_args()

    # Load prompts grouped by tag
    grouped_prompts = load_prompts(args.prompts_file)

    # Process each tag separately
    for tag, prompts in grouped_prompts.items():
        process_tag(tag, prompts, args.output_dir, args.restart_steps, args.refinement_step)

    print("\n✅ Batch Processing Complete for all tags!")

if __name__ == "__main__":
    main()