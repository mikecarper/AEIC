import sys
sys.path.append("..")

import time
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from base_model import OneStepDiffusionDecoder
from codec.codec_trainer import PixelCodec_ME_Trainer, PixelCodec_SE_Trainer
from custom_forward import MyUNet2DConditionModel_SD_forward_dual


class AEIC_ME_Trainer(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)
        self.add_unet_LoRA(lora_rank_unet=args.lora_rank_unet)
        self.codec = PixelCodec_ME_Trainer()
        if args.codec_path is not None:
            self.load_AEIC_state_dict(args.codec_path)
        self.set_training_mode(args.compile_model)


    def set_training_mode(self, compile_model):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        self.unet.conv_out.requires_grad_(True)
        
        if compile_model:
            self.codec.compile()
            self.vae.decoder.compile()
            self.unet.compile()


    def load_AEIC_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        pretrained_unet_keys = pretrained_weights["state_dict_unet"].keys()
        for k in unet_random_weights.keys():
            if "lora" in k or "conv_in" in k or "conv_out" in k:
                if k not in pretrained_unet_keys:
                    raise KeyError(f"Expecting unet key: {k} not found in provided state_dict_unet")
                unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)


    def forward(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.codec(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = self.unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "conv_out" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        torch.save(sd, outf)
    

class AEIC_SE_Trainer_S1(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)

        self.add_unet_LoRA(lora_rank_unet=args.lora_rank_unet)

        self.teacher_codec = PixelCodec_ME_Trainer()
        self.student_codec = PixelCodec_SE_Trainer()
        self.load_AEIC_ME_state_dict(args.teacher_path)
        if args.codec_path is not None:
            self.load_AEIC_SE_state_dict(args.codec_path)

        teacher_ch_list = [320, 160, 320, 320]
        student_ch_list = [256, 128, 256, 256]
        self.project = nn.ModuleList([nn.Conv2d(student_ch_list[i], teacher_ch_list[i], 1) for i in range(len(teacher_ch_list))])
        
        self.set_training_mode(args.compile_model)


    def set_training_mode(self, compile_model):
        self.unet.train()
        self.vae.train()
        self.teacher_codec.train()
        self.student_codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.teacher_codec.requires_grad_(False)
        self.student_codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        self.unet.conv_out.requires_grad_(True)
        
        if compile_model:
            self.teacher_codec.compile()
            self.student_codec.compile()
            self.vae.decoder.compile()
            self.unet.compile()


    def load_AEIC_ME_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.teacher_codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.teacher_codec.load_state_dict(codec_random_weights)


    def load_AEIC_SE_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.student_codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.student_codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        pretrained_unet_keys = pretrained_weights["state_dict_unet"].keys()
        for k in unet_random_weights.keys():
            if "lora" in k or "conv_in" in k or "conv_out" in k:
                if k not in pretrained_unet_keys:
                    raise KeyError(f"Expecting unet key: {k} not found in provided state_dict_unet")
                unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)


    def forward(self, x, ori_h=None, ori_w=None, distillation=True):

        if distillation:

            l_T, RateLossOutput, residual, student_feature = self.student_codec.forward_with_feature(x, ori_h=ori_h, ori_w=ori_w)
            l_0 = self.unet(l_T) + residual
            x_hat = self.vae.decoder(l_0).clamp(-1, 1)

            with torch.no_grad():
                teacher_feature = self.teacher_codec.get_feature(x)

            distillation_loss = 0.
            for i in range(len(student_feature)):
                fea = self.project[i](student_feature[i])
                distillation_loss += F.mse_loss(fea, teacher_feature[i].clone())
            return x_hat, RateLossOutput, distillation_loss

        else:

            if self.project is not None:
                del self.project
                self.project = None

            l_T, RateLossOutput, residual = self.student_codec(x, ori_h=ori_h, ori_w=ori_w)
            l_0 = self.unet(l_T) + residual
            x_hat = self.vae.decoder(l_0).clamp(-1, 1)
            return x_hat, RateLossOutput


    @torch.inference_mode()
    def inference(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.student_codec(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = self.unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "conv_out" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.student_codec.state_dict().items()}
        torch.save(sd, outf)


class AEIC_SE_Trainer_S2(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)

        self.add_unet_LoRA(lora_name = ["teacher_unet_lora", "student_unet_lora"], lora_rank_unet=args.lora_rank_unet)
        
        ### Setup conv_in & conv_out
        self.unet.teacher_conv_in = copy.deepcopy(self.unet.conv_in)
        self.unet.student_conv_in = copy.deepcopy(self.unet.conv_in)
        self.unet.teacher_conv_out = copy.deepcopy(self.unet.conv_out)
        self.unet.student_conv_out = copy.deepcopy(self.unet.conv_out)
        del self.unet.conv_in, self.unet.conv_out
        self.unet.forward = types.MethodType(MyUNet2DConditionModel_SD_forward_dual, self.unet)

        self.teacher_codec = PixelCodec_ME_Trainer()
        self.student_codec = PixelCodec_SE_Trainer()
        self.load_AEIC_ME_state_dict(args.teacher_path)
        self.load_AEIC_SE_state_dict(args.codec_path)

        ch_list = [320, 320, 640, 1280, 1280, 1280, 1280, 1280, 640, 320, 256]
        self.project = nn.ModuleList(nn.Conv2d(ch, ch, 1) for ch in ch_list)
        
        self.set_training_mode(args.compile_model)


    def set_training_mode(self, compile_model):
        self.unet.train()
        self.vae.train()
        self.teacher_codec.train()
        self.student_codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.teacher_codec.requires_grad_(False)
        self.student_codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "student_unet_lora" in n:
                _p.requires_grad = True
        self.unet.student_conv_in.requires_grad_(True)
        self.unet.student_conv_out.requires_grad_(True)
        
        if compile_model:
            self.teacher_codec.compile()
            self.student_codec.compile()
            self.vae.decoder.compile()
            self.unet.compile()


    def load_AEIC_ME_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.teacher_codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.teacher_codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        for k in unet_random_weights.keys():
            if "teacher_unet_lora" in k:
                revised_k = k.replace("teacher_unet_lora", "default")
            elif "teacher_conv_in" in k:
                revised_k = k.replace("teacher_conv_in", "conv_in")
            elif "teacher_conv_out" in k:
                revised_k = k.replace("teacher_conv_out", "conv_out")
            else:
                continue
            unet_random_weights[k] = pretrained_weights["state_dict_unet"][revised_k]
        self.unet.load_state_dict(unet_random_weights)


    def load_AEIC_SE_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.student_codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.student_codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        for k in unet_random_weights.keys():
            if "student_unet_lora" in k:
                revised_k = k.replace("student_unet_lora", "default")
            elif "student_conv_in" in k:
                revised_k = k.replace("student_conv_in", "conv_in")
            elif "student_conv_out" in k:
                revised_k = k.replace("student_conv_out", "conv_out")
            else:
                continue
            unet_random_weights[k] = pretrained_weights["state_dict_unet"][revised_k]
            # if "student_unet_lora" in k or "student_conv_in" in k or "student_conv_out" in k:
                # unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)


    def forward(self, x, ori_h=None, ori_w=None, distillation=True):

        if distillation:
            
            self.unet.set_adapter(["student_unet_lora"])
            l_T, RateLossOutput, residual = self.student_codec(x, ori_h=ori_h, ori_w=ori_w)
            l_0, student_feature = self.unet(l_T, "student")
            student_feature.append(residual)
            l_0 = l_0 + residual
            x_hat = self.vae.decoder(l_0).clamp(-1, 1)

            with torch.no_grad():
                self.unet.set_adapter(["teacher_unet_lora"])
                l_T, _, residual = self.teacher_codec(x, ori_h=ori_h, ori_w=ori_w)
                _, teacher_feature = self.unet(l_T, "teacher")
                teacher_feature.append(residual)

            distillation_loss = 0.
            for i in range(len(student_feature)-1):
                fea = self.project[i](student_feature[i])
                distillation_loss += F.mse_loss(fea, teacher_feature[i].clone())
            distillation_loss += F.mse_loss(student_feature[-1], teacher_feature[-1].clone())
            return x_hat, RateLossOutput, distillation_loss

        else:

            if self.project is not None:
                del self.project
                self.project = None
                self.unet.set_adapter(["student_unet_lora"])

            l_T, RateLossOutput, residual = self.student_codec(x, ori_h=ori_h, ori_w=ori_w)
            l_0, _ = self.unet(l_T, "student")
            l_0 = l_0 + residual
            x_hat = self.vae.decoder(l_0).clamp(-1, 1)
            return x_hat, RateLossOutput


    @torch.inference_mode()
    def inference(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.student_codec(x, ori_h=ori_h, ori_w=ori_w)
        self.unet.set_adapter(["student_unet_lora"])
        l_0, _ = self.unet(l_T, "student")
        l_0 = l_0 + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "student_unet_lora" in k or "student_conv_in" in k or "student_conv_out" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.student_codec.state_dict().items()}
        torch.save(sd, outf)


class AEIC_SE_Trainer_S3(OneStepDiffusionDecoder):
    def __init__(self, sd_path=None, args=None):
        super().__init__(sd_path=sd_path, args=args)
        self.add_unet_LoRA(lora_rank_unet=args.lora_rank_unet)
        self.codec = PixelCodec_SE_Trainer()
        if args.codec_path is not None:
            self.load_AEIC_SE_state_dict(args.codec_path)
        self.set_training_mode(args.compile_model)


    def set_training_mode(self, compile_model):
        self.unet.train()
        self.vae.train()
        self.codec.train()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.codec.requires_grad_(True)

        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        self.unet.conv_out.requires_grad_(True)
        
        if compile_model:
            self.codec.compile()
            self.vae.decoder.compile()
            self.unet.compile()


    def load_AEIC_SE_state_dict(self, aeic_state_dict_path):

        pretrained_weights = torch.load(aeic_state_dict_path, map_location="cpu")

        codec_random_weights = self.codec.state_dict()
        pretrained_codec_keys = pretrained_weights["state_dict_codec"].keys()
        for k in codec_random_weights.keys():
            if k not in pretrained_codec_keys:
                raise KeyError(f"Expecting codec key: {k} not found in provided state_dict_codec")
            codec_random_weights[k] = pretrained_weights["state_dict_codec"][k]
        self.codec.load_state_dict(codec_random_weights)

        unet_random_weights = self.unet.state_dict()
        pretrained_unet_keys = pretrained_weights["state_dict_unet"].keys()
        for k in unet_random_weights.keys():
            if k not in pretrained_unet_keys:
                if "default" in k:
                    revised_k = k.replace("default", "student_unet_lora")
                elif "conv_in" in k:
                    revised_k = k.replace("conv_in", "student_conv_in")
                elif "conv_out" in k:
                    revised_k = k.replace("conv_out", "student_conv_out")
                else:
                    continue
                if revised_k not in pretrained_unet_keys:
                    raise KeyError(f"Expecting unet key: {k} or {revised_k} not found in provided state_dict_unet")
                unet_random_weights[k] = pretrained_weights["state_dict_unet"][revised_k]
            else:
                unet_random_weights[k] = pretrained_weights["state_dict_unet"][k]
        self.unet.load_state_dict(unet_random_weights)


    def forward(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.codec(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = cp.checkpoint(self.unet, l_T) + residual
        x_hat = cp.checkpoint(self.vae.decoder, l_0).clamp(-1, 1)
        return x_hat, RateLossOutput


    @torch.inference_mode()
    def inference(self, x, ori_h=None, ori_w=None):
        l_T, RateLossOutput, residual = self.codec(x, ori_h=ori_h, ori_w=ori_w)
        l_0 = self.unet(l_T) + residual
        x_hat = self.vae.decoder(l_0).clamp(-1, 1)
        return x_hat, RateLossOutput

    
    def save_model(self, outf):
        sd = {}
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k or "conv_out" in k}
        sd["state_dict_codec"] = {k: v for k, v in self.codec.state_dict().items()}
        torch.save(sd, outf)