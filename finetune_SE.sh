#!/bin/bash

accelerate launch --num_processes=2 --gpu_ids="0,1," src/finetune_SE.py \
    --config="config/finetune_AEIC_SE.yaml" \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --vae_decoder_path="<PATH_TO_VAE_DECODER>/halfDecoder.ckpt" \
    --train_dataset_2K="<PATH_TO_DATASET>/dataset_2K.hdf5" \
    --codec_path="<PATH_TO_CODEC>/AEIC_SE_pretrain_200000.pkl" \
    --teacher_path="<PATH_TO_TEACHER>/AEIC_ME_ft2.pkl" \
    --test_dataset="<PATH_TO_DATASET>/Kodak" \
    --output_dir="<PATH_TO_OUTPUT_DIR>"