accelerate launch /users/Etu2/21213002/prat/diffusers/examples/unconditional_image_generation/train_unconditional.py \
   --train_data_dir /users/Etu2/21213002/CrowdsourcingDataset-Amgadetal2019/crop/128x128 \
   --resolution=64 --center_crop --random_flip \
   --output_dir="/users/Etu2/21213002/prat/Diffusion-for-Pathology/dm1" \
   --train_batch_size=16 \
   --num_epochs=500 \
   --gradient_accumulation_steps=1 \
   --learning_rate=1e-4 \
   --lr_warmup_steps=500 \
   --mixed_precision="no"