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
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from AEIC_trainer import AEIC_SE_Trainer_S2, AEIC_SE_Trainer_S3
from my_utils.training_utils import *
from oc_losses.oc_ea_dists_loss import OCEADISTSLoss
import vision_aided_loss


def train_s2(args, cfg, sd_path):
    """
    Train AEIC-SE Stage 2 (S2): Dual-side distillation at 512 resolution
    Returns: checkpoint path for S3 initialization
    """
    # S2 specific configuration
    s2_cfg = cfg["s2"]
    
    # Initialize Accelerator with S2 gradient accumulation
    # Enable find_unused_parameters for debugging DDP issues
    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=s2_cfg["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs]
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("STAGE S2: Dual UNet Distillation (512x512)")
        print("="*80)
        print(f"Max steps: {s2_cfg['max_train_steps']}")
        print(f"Patch size: {s2_cfg['train_patch_size']}")
        print(f"Batch size per GPU: {s2_cfg['train_batch_size_per_gpu']}")
        print(f"Gradient accumulation: {s2_cfg['gradient_accumulation_steps']}")
        print(f"Effective batch size: {accelerator.num_processes * s2_cfg['train_batch_size_per_gpu'] * s2_cfg['gradient_accumulation_steps']}")
        print("="*80 + "\n")
    
    expected_total = accelerator.num_processes * s2_cfg['train_batch_size_per_gpu'] * s2_cfg['gradient_accumulation_steps']
    if s2_cfg['train_batch_size_total'] != expected_total:
        raise ValueError(f"S2 batch size mismatch: config says {s2_cfg['train_batch_size_total']}, but calculated {expected_total}")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(s2_cfg['train_patch_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    train_dataset = H5Dataset(args.train_dataset_2K, transform=train_transform)
    test_dataset = ImageFolder(args.test_dataset, split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=s2_cfg['train_batch_size_per_gpu'],
        num_workers=cfg['common']['dataloader_num_workers'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg['common']['dataloader_num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    net = AEIC_SE_Trainer_S2(sd_path=sd_path, args=args)
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    if is_xformers_available():
        net.unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available, please install it")
    
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    net_disc = vision_aided_loss.Discriminator(
        cv_type='dinov3',
        output_type='conv_multi_level',
        loss_type="multilevel_sigmoid_s",
        device="cuda"
    )
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()
    
    net_dists = OCEADISTSLoss(device=accelerator.device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    net_clip = CLIPLoss().cuda()
    mse_loss = torch.nn.MSELoss()
    
    layers_to_opt = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(layers_to_opt, lr=float(s2_cfg['base']['lr']))
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5)
    
    net, net_disc, optimizer, optimizer_disc, train_dataloader = accelerator.prepare(net, net_disc, optimizer, optimizer_disc, train_dataloader)
    net_dists, alex_lpips, net_clip = accelerator.prepare(net_dists, alex_lpips, net_clip)
    
    unwrapped_net_for_eval = accelerator.unwrap_model(net)
    
    weight_dtype = torch.float32
    net.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_dists.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    stage_manager = StageManager(s2_cfg['stages'])
    progress_bar = tqdm(range(0, s2_cfg['max_train_steps']), initial=0, desc="S2 Steps", disable=not accelerator.is_local_main_process)
    
    current_hp = {}
    for k, v in cfg['common'].items():
        if k.startswith('lambda_'):
            current_hp[k] = float(v) if isinstance(v, (int, float)) else v
    for k, v in s2_cfg['base'].items():
        current_hp[k] = float(v) if isinstance(v, (int, float)) else v
    full_exp_name = cfg['exp_name'] + str(int(float(current_hp['lambda_rate'])))
    
    projection_cleaned = False
    global_step = 0
    
    while global_step < s2_cfg['max_train_steps']:
        for batch in train_dataloader:
            
            # Check stage switch
            stage, change_stage = stage_manager.get_stage(global_step)
            if change_stage:
                prev_lambda_distill = current_hp.get('lambda_distill', 0.5)
                
                for k, v in stage.items():
                    if k in ['name', 'end_step']:
                        current_hp[k] = v
                    else:
                        current_hp[k] = float(v) if isinstance(v, (int, float, str)) and k != 'name' else v
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = float(current_hp['lr'])  # Ensure lr is float
                
                if accelerator.is_main_process:
                    print(f"[S2 Stage Switch] Step {global_step} -> {current_hp}")
                
                # Clean projection layers when turning off distillation
                if current_hp['lambda_distill'] == 0 and prev_lambda_distill > 0 and not projection_cleaned:
                    if accelerator.is_main_process:
                        print(f"[S2 Projection Cleanup] Removing projection layers at step {global_step}")
                        print(f"[S2 Projection Cleanup] Reinitializing discriminator")
                    
                    if hasattr(accelerator.unwrap_model(net), 'project') and accelerator.unwrap_model(net).project is not None:
                        unwrapped_net = accelerator.unwrap_model(net)
                        del unwrapped_net.project
                        unwrapped_net.project = None
                        
                        if accelerator.is_main_process:
                            print(f"[S2 Projection Cleanup] Projection layers removed")
                        
                        accelerator.wait_for_everyone()
                        
                        layers_to_opt = [p for p in unwrapped_net.parameters() if p.requires_grad]
                        optimizer = torch.optim.AdamW(layers_to_opt, lr=float(current_hp['lr']))
                        del net_disc, optimizer_disc
                        
                        net_disc = vision_aided_loss.Discriminator(
                            cv_type='dinov3',
                            output_type='conv_multi_level',
                            loss_type="multilevel_sigmoid_s",
                            device="cuda"
                        )
                        net_disc = net_disc.cuda()
                        net_disc.requires_grad_(True)
                        net_disc.cv_ensemble.requires_grad_(False)
                        net_disc.train()
                        optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5)
                        
                        net, net_disc, optimizer, optimizer_disc = accelerator.prepare(
                            unwrapped_net, net_disc, optimizer, optimizer_disc
                        )
                        
                        net.to(accelerator.device, dtype=weight_dtype)
                        net_disc.to(accelerator.device, dtype=weight_dtype)
                        for name, module in net_disc.named_modules():
                            if "attn" in name:
                                module.fused_attn = False
                        
                        unwrapped_net_for_eval = accelerator.unwrap_model(net)
                        
                        if accelerator.is_main_process:
                            print(f"[S2 Projection Cleanup] Model and optimizer re-wrapped with DDP")
                            print(f"[S2 Projection Cleanup] Discriminator reinitialized")
                            print(f"[S2 Projection Cleanup] Skipping current batch to allow DDP to stabilize")
                    
                    projection_cleaned = True
                    
                    # CRITICAL: Skip the current batch after projection cleanup.This allows the new DDP wrapper to properly initialize before processing data
                    continue
            
            l_acc = [net, net_disc]
            
            with accelerator.accumulate(*l_acc):
                if current_hp['lambda_distill'] > 0:
                    x_hat, RateLossOutput, distillation_loss = net(
                        batch, ori_h=s2_cfg['train_patch_size'], ori_w=s2_cfg['train_patch_size'], distillation=True
                    )
                else:
                    x_hat, RateLossOutput = net(
                        batch, ori_h=s2_cfg['train_patch_size'], ori_w=s2_cfg['train_patch_size'], distillation=False
                    )
                
                x = batch.detach().float()
                x_hat = x_hat.float()
                
                loss_l2 = mse_loss(x_hat, x)
                loss_dists = net_dists(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = net_disc(x_hat, for_G=True).mean()
                
                loss = RateLossOutput.rate_loss * current_hp['lambda_rate'] \
                     + loss_l2 * current_hp['lambda_l2'] \
                     + loss_dists * current_hp['lambda_dists'] \
                     + loss_clip * current_hp['lambda_clip'] \
                     + loss_adv * current_hp['lambda_adv']
                
                if current_hp['lambda_distill'] > 0:
                    loss = loss + distillation_loss * current_hp['lambda_distill']
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, cfg['common']['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                
                loss_real = net_disc(x.detach(), for_real=True).mean() * current_hp['lambda_adv']
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg['common']['max_grad_norm'])
                optimizer_disc.step()
                optimizer_disc.zero_grad()
                
                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * current_hp['lambda_adv']
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg['common']['max_grad_norm'])
                optimizer_disc.step()
                optimizer_disc.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                
                if accelerator.is_main_process:
                    if global_step % cfg['common']['checkpointing_steps'] == 0 and global_step > 0:
                        outf = os.path.join(args.output_dir, "checkpoints",
                                          f"{full_exp_name}_s2_{global_step}.pkl")
                        accelerator.unwrap_model(net).save_model(outf)
                        print(f"[S2 Checkpoint] Saved at step {global_step}")
                    
                    if global_step % cfg['common']['eval_freq'] == 0:
                        progress_bar.update(cfg['common']['eval_freq'])
                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        val_count = 0
                        
                        for id, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(accelerator.device)
                            B, C, H, W = batch_val.shape
                            assert B == 1
                            
                            with torch.no_grad():
                                x_hat, RateLossOutput = unwrapped_net_for_eval.inference(
                                    batch_val, ori_h=H, ori_w=W
                                )
                                
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
                            
                            if val_count < cfg['common']['save_num']:
                                x = x.cpu().detach()
                                x_hat = x_hat.cpu().detach()
                                combined = torch.cat([x, x_hat], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                outf = os.path.join(args.output_dir, "eval", f"s2_val_{id}.png")
                                output_pil.save(outf)
                                val_count += 1
                        
                        logs = {}
                        assert len(l_psnr) == 24
                        logs['R'] = np.mean(l_rate)
                        logs['Y'] = np.mean(l_y)
                        logs['Z'] = np.mean(l_z)
                        logs['PSNR'] = np.mean(l_psnr)
                        logs['LPIPS'] = np.mean(l_lpips)
                        formatted_logs = {
                            'R': f"{logs['R']:.4f}",
                            'Y': f"{logs['Y']:.4f}",
                            'Z': f"{logs['Z']:.4f}",
                            'PSNR': f"{logs['PSNR']:.2f}",
                            'LPIPS': f"{logs['LPIPS']:.3f}",
                        }
                        progress_bar.set_postfix(**formatted_logs)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)
            
            if global_step >= s2_cfg['max_train_steps']:
                break
    
    # Save final S2 checkpoint for S3 initialization
    s2_final_checkpoint = os.path.join(
        args.output_dir, "checkpoints",
        f"{full_exp_name}_s2_final_{global_step}.pkl"
    )
    
    if accelerator.is_main_process:
        print(f"\n[S2 Complete] Final checkpoint saved: {s2_final_checkpoint}")
        print(f"[S2 Complete] Total steps: {global_step}")
        print(f"\n[S2 Memory] Before cleanup (Rank {accelerator.process_index}):")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        print("="*80 + "\n")
    
    # Save model first
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(net).save_model(s2_final_checkpoint)
    accelerator.wait_for_everyone()
    
    return s2_final_checkpoint


def train_s3(args, cfg, sd_path, s2_checkpoint):
    """
    Train AEIC-SE Stage 3 (S3): Higher resolution finetuning at 1024
    """
    s3_cfg = cfg['s3']
    
    accelerator = Accelerator(
        gradient_accumulation_steps=s3_cfg['gradient_accumulation_steps']
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("STAGE S3: Higher Resolution Training (1024x1024)")
        print("="*80)
        print(f"Max steps: {s3_cfg['max_train_steps']}")
        print(f"Patch size: {s3_cfg['train_patch_size']}")
        print(f"Batch size per GPU: {s3_cfg['train_batch_size_per_gpu']}")
        print(f"Gradient accumulation: {s3_cfg['gradient_accumulation_steps']}")
        print(f"Effective batch size: {accelerator.num_processes * s3_cfg['train_batch_size_per_gpu'] * s3_cfg['gradient_accumulation_steps']}")
        print(f"Loading from S2 checkpoint: {s2_checkpoint}")
        print("="*80 + "\n")
    
    expected_total = accelerator.num_processes * s3_cfg['train_batch_size_per_gpu'] * s3_cfg['gradient_accumulation_steps']
    if s3_cfg['train_batch_size_total'] != expected_total:
        raise ValueError(f"S3 batch size mismatch: config says {s3_cfg['train_batch_size_total']}, but calculated {expected_total}")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(s3_cfg['train_patch_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    train_dataset = H5Dataset(args.train_dataset_2K, transform=train_transform, res=s3_cfg['train_patch_size'])
    test_dataset = ImageFolder(args.test_dataset, split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=s3_cfg['train_batch_size_per_gpu'],
        num_workers=cfg['common']['dataloader_num_workers'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg['common']['dataloader_num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    
    args_s3 = type('Args', (), {})()
    for attr in dir(args):
        if not attr.startswith('_'):
            setattr(args_s3, attr, getattr(args, attr))
    args_s3.codec_path = s2_checkpoint
    
    net = AEIC_SE_Trainer_S3(sd_path=sd_path, args=args_s3)
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    if is_xformers_available():
        net.unet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available, please install it")
    
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    net_disc = vision_aided_loss.Discriminator(
        cv_type='dinov3',
        output_type='conv_multi_level',
        loss_type="multilevel_sigmoid_s",
        device="cuda"
    )
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()
    
    net_dists = OCEADISTSLoss(device=accelerator.device)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    net_clip = CLIPLoss().cuda()
    mse_loss = torch.nn.MSELoss()
    
    layers_to_opt = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(layers_to_opt, lr=float(s3_cfg['base']['lr']))
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5)
    
    net, net_disc, optimizer, optimizer_disc, train_dataloader = accelerator.prepare(net, net_disc, optimizer, optimizer_disc, train_dataloader)
    net_dists, alex_lpips, net_clip = accelerator.prepare(net_dists, alex_lpips, net_clip)
    
    weight_dtype = torch.float32
    net.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_dists.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)
    
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    
    stage_manager = StageManager(s3_cfg['stages'])
    progress_bar = tqdm(range(0, s3_cfg['max_train_steps']), initial=0, desc="S3 Steps",
                       disable=not accelerator.is_local_main_process)
    
    current_hp = {}
    for k, v in cfg['common'].items():
        if k.startswith('lambda_'):
            current_hp[k] = float(v) if isinstance(v, (int, float)) else v
    for k, v in s3_cfg['base'].items():
        current_hp[k] = float(v) if isinstance(v, (int, float)) else v
    full_exp_name = cfg['exp_name'] + str(int(float(current_hp['lambda_rate'])))
    
    global_step = 0
    s2_steps = cfg['s2']['max_train_steps']  # For logging absolute step numbers
    
    # Training loop
    while global_step < s3_cfg['max_train_steps']:
        for batch in train_dataloader:
            
            stage, change_stage = stage_manager.get_stage(global_step)
            if change_stage:
                for k, v in stage.items():
                    if k in ['name', 'end_step']:
                        current_hp[k] = v
                    else:
                        current_hp[k] = float(v) if isinstance(v, (int, float, str)) and k != 'name' else v
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = float(current_hp['lr'])  # Ensure lr is float
                
                if accelerator.is_main_process:
                    print(f"[S3 Stage Switch] Step {global_step} (abs: {s2_steps + global_step}) -> {current_hp}")
            
            l_acc = [net, net_disc]
            
            with accelerator.accumulate(*l_acc):
                x_hat, RateLossOutput = net(batch, ori_h=s3_cfg['train_patch_size'], ori_w=s3_cfg['train_patch_size'])
                
                x = batch.detach().float()
                x_hat = x_hat.float()
                
                loss_l2 = mse_loss(x_hat, x)
                loss_dists = net_dists(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = net_disc(x_hat, for_G=True).mean()
                
                loss = RateLossOutput.rate_loss * current_hp['lambda_rate'] \
                     + loss_l2 * current_hp['lambda_l2'] \
                     + loss_dists * current_hp['lambda_dists'] \
                     + loss_clip * current_hp['lambda_clip'] \
                     + loss_adv * current_hp['lambda_adv']
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, cfg['common']['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                
                loss_real = net_disc(x.detach(), for_real=True).mean() * current_hp['lambda_adv']
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg['common']['max_grad_norm'])
                optimizer_disc.step()
                optimizer_disc.zero_grad()
                
                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * current_hp['lambda_adv']
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), cfg['common']['max_grad_norm'])
                optimizer_disc.step()
                optimizer_disc.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                absolute_step = s2_steps + global_step
                
                if accelerator.is_main_process:
                    if global_step % cfg['common']['checkpointing_steps'] == 0 and global_step > 0:
                        outf = os.path.join(args.output_dir, "checkpoints",
                                          f"{full_exp_name}_s3_{absolute_step}.pkl")
                        accelerator.unwrap_model(net).save_model(outf)
                        print(f"[S3 Checkpoint] Saved at step {global_step} (abs: {absolute_step})")
                    
                    if global_step % cfg['common']['eval_freq'] == 0:
                        progress_bar.update(cfg['common']['eval_freq'])
                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        val_count = 0
                        
                        for id, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(accelerator.device)
                            B, C, H, W = batch_val.shape
                            assert B == 1
                            
                            with torch.no_grad():
                                x_hat, RateLossOutput = accelerator.unwrap_model(net).inference(
                                    batch_val, ori_h=H, ori_w=W
                                )
                                
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
                            
                            if val_count < cfg['common']['save_num']:
                                x = x.cpu().detach()
                                x_hat = x_hat.cpu().detach()
                                combined = torch.cat([x, x_hat], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                outf = os.path.join(args.output_dir, "eval", f"s3_val_{id}.png")
                                output_pil.save(outf)
                                val_count += 1
                        
                        logs = {}
                        assert len(l_psnr) == 24
                        logs['R'] = np.mean(l_rate)
                        logs['Y'] = np.mean(l_y)
                        logs['Z'] = np.mean(l_z)
                        logs['PSNR'] = np.mean(l_psnr)
                        logs['LPIPS'] = np.mean(l_lpips)
                        formatted_logs = {
                            'R': f"{logs['R']:.4f}",
                            'Y': f"{logs['Y']:.4f}",
                            'Z': f"{logs['Z']:.4f}",
                            'PSNR': f"{logs['PSNR']:.2f}",
                            'LPIPS': f"{logs['LPIPS']:.3f}",
                        }
                        progress_bar.set_postfix(**formatted_logs)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=absolute_step)
            
            if global_step >= s3_cfg['max_train_steps']:
                break
    
    if accelerator.is_main_process:
        s3_final_checkpoint = os.path.join(
            args.output_dir, "checkpoints",
            f"{full_exp_name}_s3_final_{s2_steps + global_step}.pkl"
        )
        accelerator.unwrap_model(net).save_model(s3_final_checkpoint)
        print(f"\n[S3 Complete] Final checkpoint saved: {s3_final_checkpoint}")
        print(f"[S3 Complete] Total steps: {global_step} (absolute: {s2_steps + global_step})")
        print("="*80 + "\n")
    
    accelerator.wait_for_everyone()
    
    del net, net_disc, optimizer, optimizer_disc, train_dataloader, test_dataloader
    del net_dists, alex_lpips, net_clip, accelerator
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    """
    Main entry point for AEIC-SE finetuning
    Executes S2 → S3 training pipeline
    """
    if args.teacher_path is None:
        raise ValueError("--teacher_path is required for AEIC-SE finetune (Stage 2). Please provide path to AEIC-ME model.")
    if args.codec_path is None:
        raise ValueError("--codec_path is required for AEIC-SE finetune. Please provide path to AEIC-SE Stage 1 model.")
    
    cfg = load_config(args.config)
    
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    print("\n" + "="*80)
    print("AEIC-SE FINETUNING PIPELINE")
    print("="*80)
    print(f"Experiment: {cfg['exp_name']}")
    print(f"Output directory: {args.output_dir}")
    print(f"S2 max steps: {cfg['s2']['max_train_steps']}")
    print(f"S3 max steps: {cfg['s3']['max_train_steps']}")
    print(f"Total steps: {cfg['s2']['max_train_steps'] + cfg['s3']['max_train_steps']}")
    print("="*80 + "\n")
    
    # Execute S2 training
    s2_checkpoint = train_s2(args, cfg, sd_path)
    
    
    print("\n" + "="*80)
    print("🧹 Aggressive memory cleanup...")
    print("="*80)
    
    # Phase 1: Python garbage collection (S2 locals already destroyed by return)
    print("Phase 1: Python garbage collection...")
    for i in range(10):
        collected = gc.collect()
        if collected > 0:
            print(f"  GC round {i+1}: collected {collected} objects")
    
    # Phase 2: CUDA synchronization
    print("Phase 2: CUDA synchronization...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Phase 3: Empty CUDA cache
    print("Phase 3: Emptying CUDA cache...")
    for _ in range(5):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    
    # Phase 4: Reset CUDA memory statistics
    print("Phase 4: Resetting CUDA memory statistics...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    
    # Phase 5: IPC cleanup (multi-process)
    print("Phase 5: CUDA IPC cleanup...")
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    
    # Phase 6: Final cleanup round
    print("Phase 6: Final cleanup round...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    
    # Phase 7: Distributed synchronization
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Phase 8: Wait for cleanup to stabilize
    print("Phase 7: Waiting for cleanup to stabilize...")
    import time
    time.sleep(3)
    
    print("\n✅ NUCLEAR CLEANUP: Complete!")
    
    # Print GPU memory after cleanup
    print(f"\n[Post-S2 Memory] After cleanup:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    if allocated > 1.0 or reserved > 2.0:
        print(f"  ⚠️  WARNING: Memory not fully released!")
    else:
        print(f"  ✅ Memory successfully cleared")
    print("="*80 + "\n")
    
    # Execute S3 training
    train_s3(args, cfg, sd_path, s2_checkpoint)
    
    print("\n" + "="*80)
    print("AEIC-SE FINETUNING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    args = parse_args_training()
    main(args)
