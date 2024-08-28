import numpy as np
import torch
import random
from PIL import Image
import argparse
import os, glob, json
from diffusers import StableDiffusionPipeline
import copy
from functools import reduce
import operator
from train_erase import view_images


def edit_model(model, old_images, new_images, retain_images=None, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, technique='tensor'):
    # Get the VAE encoder layers
    encoder_layers = model.vae.encoder.down_blocks

    projection_matrices = []
    og_matrices = []
    for block in encoder_layers:
        for layer in block.attentions:
            projection_matrices.append(layer.to_q)
            projection_matrices.append(layer.to_k)
            projection_matrices.append(layer.to_v)
            og_matrices.extend([copy.deepcopy(layer.to_q), copy.deepcopy(layer.to_k), copy.deepcopy(layer.to_v)])

    # Reset parameters
    for idx_, l in enumerate(projection_matrices):
        l.weight = torch.nn.Parameter(copy.deepcopy(og_matrices[idx_].weight))

    layers_to_edit = range(len(projection_matrices)) if layers_to_edit is None else layers_to_edit

    print(f"Editing {len(old_images)} concepts")
    for layer_num in layers_to_edit:
        with torch.no_grad():
            mat1 = lamb * projection_matrices[layer_num].weight
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=model.device)

            for old_image, new_image in zip(old_images, new_images):
                old_emb = model.vae.encode(old_image).latent_dist.sample()
                new_emb = model.vae.encode(new_image).latent_dist.sample()
                
                context = old_emb.detach()
                
                if technique == 'tensor':
                    o_embs = projection_matrices[layer_num](old_emb).detach()
                    u = o_embs / o_embs.norm()
                    new_embs = projection_matrices[layer_num](new_emb).detach()
                    new_emb_proj = (u * new_embs).sum()
                    target = new_embs - (new_emb_proj) * u
                elif technique == 'replace':
                    target = projection_matrices[layer_num](new_emb).detach()
                else:
                    target = projection_matrices[layer_num](new_emb).detach()

                context_vector = context.reshape(context.shape[0], context.shape[1], -1)
                context_vector_T = context.reshape(context.shape[0], -1, context.shape[1])
                value_vector = target.reshape(target.shape[0], target.shape[1], -1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale * for_mat1
                mat2 += erase_scale * for_mat2

            if retain_images:
                for retain_image in retain_images:
                    retain_emb = model.vae.encode(retain_image).latent_dist.sample()
                    context = retain_emb.detach()
                    target = projection_matrices[layer_num](retain_emb).detach()
                    context_vector = context.reshape(context.shape[0], context.shape[1], -1)
                    context_vector_T = context.reshape(context.shape[0], -1, context.shape[1])
                    value_vector = target.reshape(target.shape[0], target.shape[1], -1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += preserve_scale * for_mat1
                    mat2 += preserve_scale * for_mat2

            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited {len(old_images)} concepts and Retained {len(retain_images) if retain_images else 0} concepts')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='TrainImageErase',
                    description='Finetuning Stable Diffusion VAE to erase image concepts')
    parser.add_argument('--source_images', help='paths to source images to erase', type=str, required=True)
    parser.add_argument('--target_images', help='paths to target images to guide towards', type=str, required=True)
    parser.add_argument('--preserve_images', help='paths to images to preserve', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--device', help='cuda device to train on', type=str, required=False, default='cuda:0')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=0.1)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1.0)
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load Stable Diffusion model
    sd14 = "CompVis/stable-diffusion-v1-4"
    sd21 = 'stabilityai/stable-diffusion-2-1-base'
    model_version = sd14 if args.base == '1.4' else sd21
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version).to(device)

    # Prepare image transformation
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512))
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(device)
        return image

    # Load and preprocess images
    def load_images(image_paths):
        return [preprocess_image(path) for path in image_paths.split(',')]

    source_images = load_images(args.source_images)
    target_images = load_images(args.target_images)
    preserve_images = load_images(args.preserve_images) if args.preserve_images else None

    assert len(source_images) == len(target_images), "Number of source and target images must be the same"

    # Edit model
    ldm_stable = edit_model(ldm_stable, source_images, target_images, retain_images=preserve_images,
                            technique=args.technique, erase_scale=args.erase_scale, preserve_scale=args.preserve_scale)

    # Save model
    torch.save(ldm_stable.vae.state_dict(), f'models/erased_sd_vae_{args.technique}.pt')

    # Save concept information
    concept_info = {
        'source_images': args.source_images,
        'target_images': args.target_images,
        'preserve_images': args.preserve_images,
        'technique': args.technique,
        'erase_scale': args.erase_scale,
        'preserve_scale': args.preserve_scale,
        'base_model': model_version
    }
    with open(f'info/erased_sd_vae_{args.technique}.json', 'w') as fp:
        json.dump(concept_info, fp)

    print(f"Model and info saved with technique: {args.technique}")