import os
import gc
import lpips
import numpy as np
import transformers
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm.auto import tqdm
from accelerate import Accelerator
from compressai.datasets import ImageFolder

import torch
import torch.utils.checkpoint
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from AEIC_trainer import AEIC_SE_Trainer_S1
from my_utils.training_utils import *

        
def main(args):

    # Validate required arguments for AEIC-SE Stage 1 training
    if args.teacher_path is None:
        raise ValueError("--teacher_path is required for AEIC-SE pretrain (Stage 1). Please provide the path to pretrained AEIC-ME model.")

    cfg = load_config(args.config)
    stage_manager = StageManager(cfg["stages"])

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    accelerator = Accelerator(gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
    assert cfg["train_batch_size_total"] == accelerator.num_processes * cfg["train_batch_size_per_gpu"] * cfg["gradient_accumulation_steps"]
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    train_dataset = H5Dataset(
        args.train_dataset_2K,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(cfg["train_patch_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataset_aux = H5Dataset(
        args.train_dataset_LSDIR,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(cfg["train_patch_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataset = ConcatDataset([train_dataset, train_dataset_aux])
    test_dataset = ImageFolder(args.test_dataset, split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["train_batch_size_per_gpu"],
        num_workers=cfg["dataloader_num_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg["dataloader_num_workers"],
        shuffle=False,
        pin_memory=True,
    )

    net = AEIC_SE_Trainer_S1(sd_path=sd_path, args=args)

    if accelerator.is_main_process:
        main_device = accelerator.device
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    if is_xformers_available():
        # net.vae.enable_xformers_memory_efficient_attention()
        net.unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    net_clip = CLIPLoss().cuda()

    layers_to_opt = []
    for p in net.parameters():
        if p.requires_grad:
            layers_to_opt.append(p)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=1e-4)
    
    net, optimizer, train_dataloader, net_lpips, alex_lpips, net_clip = accelerator.prepare(net, optimizer, train_dataloader, net_lpips, alex_lpips, net_clip)
 
    weight_dtype = torch.float32
    net.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    progress_bar = tqdm(range(0, cfg["max_train_steps"]), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
    mse_loss = torch.nn.MSELoss()

    global_step = 0
    current_hp = {}
    for k, v in cfg["base"].items():
        current_hp[k] = float(v)
    full_exp_name = cfg["exp_name"]
    
    # Flag to track if we've cleaned projection layers
    projection_cleaned = False
    
    while global_step <= cfg["max_train_steps"]:
        for batch in train_dataloader:

            stage, change_stage = stage_manager.get_stage(global_step)
            if change_stage:
                for k, v in stage.items():
                    if k in ["name", "end_step"]:
                        current_hp[k] = v
                    else:
                        current_hp[k] = float(v)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_hp["lr"]
                if accelerator.is_main_process:
                    print(f"[Stage Switch] Step {global_step} -> {current_hp}")
                
                # CRITICAL: Clean projection layers when turning off distillation
                # This prevents "unused parameter" errors in DDP training
                if current_hp["lambda_distill"] == 0 and not projection_cleaned:
                    if accelerator.is_main_process:
                        print(f"[Projection Cleanup] Removing projection layers at step {global_step}")
                        print(f"[Projection Cleanup] lambda_distill changed from >0 to 0.0")
                    
                    if hasattr(accelerator.unwrap_model(net), 'project') and accelerator.unwrap_model(net).project is not None:
                        unwrapped_net = accelerator.unwrap_model(net)
                        
                        del unwrapped_net.project
                        unwrapped_net.project = None
                        
                        if accelerator.is_main_process:
                            print(f"[Projection Cleanup] Projection layers removed from model")
                        
                        accelerator.wait_for_everyone()
                        
                        # Rebuild optimizer without projection parameters
                        layers_to_opt = []
                        for p in unwrapped_net.parameters():
                            if p.requires_grad:
                                layers_to_opt.append(p)
                        optimizer = torch.optim.AdamW(layers_to_opt, lr=current_hp["lr"])
                        
                        net = accelerator.prepare(unwrapped_net)
                        optimizer = accelerator.prepare(optimizer)
                        
                        if accelerator.is_main_process:
                            print(f"[Projection Cleanup] Model and optimizer re-wrapped with DDP")
                            print(f"[Projection Cleanup] Optimizer rebuilt with {len(layers_to_opt)} parameters")
                    
                    projection_cleaned = True
            
            l_acc = [net]

            with accelerator.accumulate(*l_acc):
                if current_hp["lambda_distill"] > 0:
                    x_hat, RateLossOutput, distillation_loss = net(batch, ori_h=cfg["train_patch_size"], ori_w=cfg["train_patch_size"], distillation=True)
                else:
                    x_hat, RateLossOutput = net(batch, ori_h=cfg["train_patch_size"], ori_w=cfg["train_patch_size"], distillation=False)
                x = batch.detach().float()
                x_hat = x_hat.float()

                loss_l2 = mse_loss(x_hat, x)
                loss_lpips = net_lpips(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)

                loss = RateLossOutput.rate_loss * current_hp["lambda_rate"] \
                     + loss_l2 * current_hp["lambda_l2"] \
                     + loss_lpips * current_hp["lambda_lpips"] \
                     + loss_clip * current_hp["lambda_clip"]

                if current_hp["lambda_distill"] > 0:
                    loss = loss + distillation_loss * current_hp["lambda_distill"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg["checkpointing_steps"] == 0 and global_step > 0:
                        outf = os.path.join(args.output_dir, "checkpoints", f"{full_exp_name}_{global_step}.pkl")
                        accelerator.unwrap_model(net).save_model(outf)
                        print(f"save {global_step}")

                    if global_step % cfg["eval_freq"] == 0:
                        progress_bar.update(cfg["eval_freq"])
                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        val_count = 0

                        for id, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(main_device)
                            B, C, H, W = batch_val.shape
                            assert B == 1, "Use batch size 1 for eval."

                            with torch.no_grad():
                                x_hat, RateLossOutput = accelerator.unwrap_model(net).inference(batch_val, ori_h=H, ori_w=W)

                                x = (batch_val * 0.5 + 0.5).float()
                                x_hat = (x_hat * 0.5 + 0.5).float()

                                loss_l2 = mse_loss(x_hat, x)
                                loss_psnr = 10 * (-torch.log(loss_l2) / np.log(10))
                                loss_lpips = alex_lpips(x_hat, x)

                                loss_R = RateLossOutput.quantized_total_bpp.detach()
                                loss_yrate = RateLossOutput.quantized_latent_bpp.detach()
                                loss_zrate = RateLossOutput.quantized_hyper_bpp.detach()

                                l_rate.append(loss_R.item())
                                l_y.append(loss_yrate.item())
                                l_z.append(loss_zrate.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(loss_psnr.item())

                            if val_count < cfg["save_num"]:
                                x = x.cpu().detach()
                                x_hat = x_hat.cpu().detach()
                                combined = torch.cat([x, x_hat], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                outf = os.path.join(args.output_dir, "eval", f"val_{id}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs = {}
                        assert len(l_psnr) == 24
                        logs["R"] = np.mean(l_rate)
                        logs["Y"] = np.mean(l_y)
                        logs["Z"] = np.mean(l_z)
                        logs["PSNR"] = np.mean(l_psnr)
                        logs["LPIPS"] = np.mean(l_lpips)
                        formatted_logs = {
                            "R": f"{logs['R']:.4f}",
                            "Y": f"{logs['Y']:.4f}",
                            "Z": f"{logs['Z']:.4f}",
                            "PSNR": f"{logs['PSNR']:.2f}",
                            "LPIPS": f"{logs['LPIPS']:.3f}",
                        }
                        progress_bar.set_postfix(**formatted_logs)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_training()
    main(args)
