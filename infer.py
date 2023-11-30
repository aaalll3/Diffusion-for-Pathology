from diffusers import DiffusionPipeline


generator = DiffusionPipeline.from_pretrained("./pretrained/dm1").to("cuda")
for i in range(100):
    image = generator().images[0]
    image.save(f"/users/Etu2/21213002/prat/Diffusion-for-Pathology/generated/uc_500_64x64/g_image_64x64_{i}.png")
