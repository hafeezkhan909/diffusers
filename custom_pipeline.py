import torch
from diffusers import StableDiffusion3Pipeline

class StableDiffusion3CustomPipeline(StableDiffusion3Pipeline):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        base_pipe = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        base_pipe.__class__ = cls  # Change class type to our custom pipeline
        return base_pipe

    def save_latents(self, prompt, num_inference_steps, guidance_scale, resume_step, save_path):
        """
        Runs inference up to `resume_step` and saves the latents at that point.
        """
        self.return_intermediate_latents = True  # ✅ Enable latent saving dynamically

        latents = self(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            return_intermediate_latents=True, 
            resume_step=resume_step
        )  # ✅ The pipeline will return latents at step 25

        print(f"Saving latents at step {resume_step}")
        torch.save({"latents": latents.clone().detach()}, save_path)

    def resume_from_latents(self, prompt, num_inference_steps, guidance_scale, latents, resume_step):
        """
        Resumes inference from a saved latent state.
        """
        image = self(
            prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            latents=latents, 
            resume_step=resume_step
        ).images[0]  # Generate final image

        return image
