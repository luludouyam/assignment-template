from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

# Load the model
model = 'lykon/dreamshaper-8-lcm'
pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16)
pipe.to("mps")

# Set the scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Infinite loop to generate landscape images
while True:
    prompt = input("Describe a landscape and press enter to generate an image:\n>>> ")
    landscape_prompt = f"A beautiful landscape of {prompt}"
    images = pipe(landscape_prompt, num_inference_steps=8, guidance_scale=1.5).images
    images[0].show()
