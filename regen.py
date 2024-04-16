import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# !pip3 install diffusers==0.11.1
# !pip3 install transformers scipy ftfy accelerate
# !pip3 install torch
# !pip3 install torch diffusers

generator = torch.Generator().manual_seed(1024)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")  
prompt = "a photograph of a woman with {d_face_shape} face shape wearing {selected_jewel} jewelery"
image = pipe(prompt, generator=generator).images[0]
image.show()