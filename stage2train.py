import argparse
import logging
import time
import itertools
from ip_adapter.attention_processornewsp3zuihao import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
import torch
import math
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import datasets
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL,  UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from ip_adapter.attention_processor import Cross_Attention

from torch import nn
from src.dataset.stage3_datasetall import collate_fn, ImageDatasetvitonhd,ImageDatasetDressCode
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from adapter.resampler import Resampler

from adapter.attention_processor3two import SAttnProcessor2_0,RefCAttnProcessor2_0,RefSAttnProcessor2_0

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--pretrained_model_stage3_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
 
    parser.add_argument(
        "--pretrained_vae_model_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="ip-adapter_sd15.bin",
        help="Path to pretrained IP-Adapter model file (e.g., ip-adapter-sd15.bin).",
    )
    parser.add_argument(
        "--dream_order",
        type=str,
        default="1",
        help="DREAM order parameter: '1' (best, 2-3x faster), '2', '3', or 'inf' (standard training). "
             "DREAM: Diffusion Rectification and Estimation-Adaptive Models (CVPR 2024)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
  
   
    parser.add_argument(
        "--empty_qwen_text_embeddings_root",
        type=str,
        default="DressCode/qwen_text_embeddings/empty_text_embedding.safetensors",
        help="Path to empty qwen text embeddings file.",
    )
    parser.add_argument(
        "--qwen_text_embeddings_root",
        type=str,
        default="DressCode/qwen_text_embeddings",
        help="Path to qwen text embeddings file.",
    )
    parser.add_argument(
        "--qwen_ipa_embeddings_root",
        type=str,
        default="DressCode/qwen_ipa_text_embeddings",
        help="Path to qwen text embeddings file.",
    )
    parser.add_argument(
        "--qwen_CA_embeddings_root",
        type=str,
        default="DressCode/qwen_ca_text_embeddings",
        help="Path to qwen text embeddings file.",
    )
    parser.add_argument(
        "--clip_embeddings_root",
        type=str,
        default="DressCode/clip_embeddings",
        help="Path to clip image embeddings file.",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="DressCode",
        help="Path to image root path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--json_path", type=str, default="DressCode/dresscode_train.json", help="json path", )
    parser.add_argument("--siglip_embeddings_root", type=str, default="DressCode/siglip_embeddings", help="siglip embeddings root", )
    parser.add_argument(
        '--clip_penultimate',
        type=bool,
        default=False,
        help='Use penultimate CLIP layer for text embedding'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0.05, help="noise_offset."
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=0, help="noise_offset."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--pretrained_adapter_model_path",
        type=str,
        default="",
        help="Path to pretrained IP-Adapter model file (e.g., ip-adapter-plus_sd15.bin).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=2000, help="Number of steps for the warmup in the lr scheduler."
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="2000",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=384,
        help="Image width.",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=512,
        help="Image height.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--stage1_output_dir", type=str, default="save_data/stage1/outpus", help="Stage1 output directory.")
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default="",
        help="Path to deepspeed config file. IMAGDressing-main/zero_stage2_config.json",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        "epoch": epoch,
        "last_global_step": last_global_step,
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f"checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}"
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(model, load_dir, tag=None, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
    The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(load_dir, tag=tag, **kwargs)
    epoch = checkpoint_state_dict["epoch"]
    last_global_step = checkpoint_state_dict["last_global_step"]
    del checkpoint_state_dict
    return (epoch, last_global_step)


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def format_time(seconds):
    """将秒数格式化为易读的时间字符串"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    elif seconds < 86400:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{hours:.0f}小时{minutes:.0f}分钟"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{days:.0f}天{hours:.0f}小时"

class HarmonyAttention(nn.Module):
    def __init__(self,
                 image_hidden_size=1280,     # Input image feature dimension
                 text_context_dim=2048,      # Input text context feature dimension
                 inter_dim=2560,             # Intermediate projection dimension
                 cross_heads=10,             # Number of cross-attention heads
                 reshape_blocks=8,           # Number of image feature blocks
                 cross_value_dim=64,         # Value dimension per head after dimensionality reduction
                 scale=1.0,                  # Output scaling factor
                 fusion_method="qformer"): # Fusion method selection: mlp, cross_attention, qformer
        super().__init__()
        
        self.scale = scale #1
        self.reshape_blocks = reshape_blocks #8
        self.cross_query_dim = inter_dim // reshape_blocks #320
        self.fusion_method = fusion_method
        self.image_hidden_size = image_hidden_size
        self.text_context_dim = text_context_dim
        
        # Image projection required by all methods
        self.fc1 = nn.Linear(image_hidden_size, inter_dim)
        print(fusion_method)
        if fusion_method == "cross_attention":
            # 1. Cross-attention
            self.fusion_text_image = Cross_Attention(
                query_dim=self.cross_query_dim,#320
                context_dim=text_context_dim,
                heads=cross_heads,#8
                value_dim=cross_value_dim#64
            )
        flattened_dim = cross_value_dim * cross_heads * reshape_blocks
        self.ln = nn.LayerNorm(flattened_dim)
        self.fc2 = nn.Linear(flattened_dim, image_hidden_size)



    def forward(self, text_embeds, image_embeds):
            """
            Args:
                text_embeds: [B, T, text_context_dim]
                image_embeds: [B, image_hidden_size]
            Returns:
                out: [B, image_hidden_size], image features fused with text conditions
            """
            B = image_embeds.size(0)

            # Map image features to intermediate dimension and reshape into multiple blocks
            x = self.fc1(image_embeds)  # [B, inter_dim]
            x = x.view(B, self.reshape_blocks, self.cross_query_dim)  # [B, N_blocks, query_dim]  

            # Cross-Attention: interaction between image blocks and text
         #   print(self.fusion_text_image)
            attended = self.fusion_text_image(x, text_embeds)  # [B, N_blocks, value_dim * heads]
        #    print(attended.shape)
            # Flatten, normalize, and project back
            attended = attended.view(B, -1)     # [B, flattened_dim]
            out = self.ln(attended)
            out = self.fc2(out) * self.scale

            return out

class SD(torch.nn.Module):
    """SD"""
    def __init__(self, unet,ref_unet,
        ref_ipa_model,
        ipa_model,
        unet_adapter_modules,
        pretrained_ip_adapter_path,
        inter_dim=2560,           # Use command-line arguments
        cross_heads=8,         # Use command-line arguments
        reshape_blocks=8,   # Use command-line arguments
        cross_value_dim=64, # Use command-line arguments
        fusion_method='cross_attention'):  # Fusion method parameter
        super().__init__()
        self.unet = unet
        self.ref_unet = ref_unet
    
        self.ref_ipa_model = ref_ipa_model
  
        self.ipa_model = ipa_model
 
        self.unet_adapter_modules = unet_adapter_modules
      #  self.unet_adapter_modules1 = unet_adapter_modules1
        # Directly use the fusion_method parameter, no need to convert to a boolean flag
        self.composed_modules = HarmonyAttention(
            image_hidden_size=1024,     # Image feature dimension fixed
            text_context_dim=1024,      # Text context dimension fixed
            inter_dim=inter_dim,
            cross_heads=cross_heads,
            reshape_blocks=reshape_blocks,
            cross_value_dim=cross_value_dim,
            scale=1.0,                  # Scaling factor fixed
            fusion_method=fusion_method  # Directly pass the fusion method parameter
        )
        
        self.load_from_checkpoint(pretrained_ip_adapter_path)
    
    def forward(self, stage2_latents_concat,
                stage3_latents_concat,
                qwen_text_embeddings,
                siglip_s_img,
                qwen_ipa_embeddings,
                qwen_CA_embeddings,
                clip_t_img,
                timesteps,
                inshop_mask,
                mask_latent):
        time_timg_t = clip_t_img
        ref_timesteps = torch.zeros_like(timesteps)
        siglip_s_img = self.ref_ipa_model(siglip_s_img)
        _ = self.ref_unet(
            stage2_latents_concat,
            ref_timesteps,
            encoder_hidden_states=None,
            return_dict=False,
            cross_attention_kwargs={
            "image_hidden_states": siglip_s_img,  
            "text_hidden_states": qwen_text_embeddings,    
             }
        )
        # get cache tensors
        sa_hidden_states = {}
        # 输入图像：宽768，高1024，VAE下采样8倍 -> latent: 宽96，高128
        # 宽度维度拼接后：宽192，高128
        H_input = 64  # VAE下采样后的高度128
        W_single = 48  # VAE下采样后的单图宽度96
        W_total_input = 96  # 拼接后的总宽度 (2*W_single)192
        
        # 存储每个layer对应的缩放后的mask
        inshop_mask_dict = {}
        mask_latent_dict = {}
        
        for name in self.ref_unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
               cached_hidden_states = self.ref_unet.attn_processors[name].cache["hidden_states"]
               B, N, C = cached_hidden_states.shape
               
               # 根据layer name推断当前层的分辨率
               if name.startswith("mid_block"):
                   # mid_block: 最小分辨率，通常是下采样3次
                   scale = 2 ** 3
               elif name.startswith("up_blocks"):
                   # up_blocks: 从后往前，block_id越大越接近输入分辨率
                   block_id = int(name.split("up_blocks.")[1].split(".")[0])
                   # up_blocks有4个，从最小分辨率开始上采样
                   # up_blocks.3是最后，接近输入；up_blocks.0是最小分辨率
                   scale = 2 ** (3 - block_id)
               elif name.startswith("down_blocks"):
                   # down_blocks: 从前往后，block_id越大分辨率越小
                   block_id = int(name.split("down_blocks.")[1].split(".")[0])
                   # down_blocks.0是输入层，down_blocks.3是最小分辨率
                   scale = 2 ** block_id
               else:
                   # 默认使用输入分辨率
                   scale = 1
               
               # 计算当前层的H和W_total
               H = H_input // scale
               W_total = W_total_input // scale
               
              
               W_single_scaled = W_single // scale
               target_size = (H, W_single_scaled)  # 高度和宽度都按scale缩放
               if inshop_mask is not None:
                   inshop_mask_scaled = torch.nn.functional.interpolate(
                       inshop_mask, size=target_size, mode="nearest"
                   )
                   # 展平为 (B, H*W) 形式，与sa_hidden_states的空间维度N对应
                   # 从 (B, C, H, W) -> (B, C*H*W) -> (B, H*W) 如果C=1
                   B_mask, C_mask, H_mask, W_mask = inshop_mask_scaled.shape
                   inshop_mask_flat = inshop_mask_scaled.flatten(start_dim=2).squeeze(1)  # (B, H*W)
                   inshop_mask_dict[name] = inshop_mask_flat
               if mask_latent is not None:
                   mask_latent_scaled = torch.nn.functional.interpolate(
                       mask_latent, size=target_size, mode="nearest"
                   )
                   # 展平为 (B, H*W) 形式
                   B_mask, C_mask, H_mask, W_mask = mask_latent_scaled.shape
                   mask_latent_flat = mask_latent_scaled.flatten(start_dim=2).squeeze(1)  # (B, H*W)
                   mask_latent_dict[name] = mask_latent_flat

                   mask_latent_expanded = mask_latent_flat.unsqueeze(-1)  # (B, H*W, 1)
           
                   cached_hidden_states = cached_hidden_states * mask_latent_expanded  # (B, H*W, C)
               
               sa_hidden_states[name] = cached_hidden_states
        
        
        composed_embeds = self.composed_modules(qwen_ipa_embeddings, clip_t_img)
        clip_t_img = clip_t_img + composed_embeds
        
        ip_tokens = self.ipa_model(clip_t_img)
        encoder_hidden_states = torch.cat([qwen_CA_embeddings, ip_tokens], dim=1)
        
        # Predict the noise residual
        noise_pred = self.unet(
        stage3_latents_concat, 
        timesteps, 
        class_labels=time_timg_t, 
        encoder_hidden_states=encoder_hidden_states,
        cross_attention_kwargs={
                "sa_hidden_states": sa_hidden_states,
                "inshop_mask": inshop_mask_dict,
                "mask_latent": mask_latent_dict,
                "train": False,

            }
        ).sample

        return noise_pred

    def load_from_checkpoint(self, pretrained_ip_adapter_path: str):
        # Calculate original checksums
        orig_ipa_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ipa_model.parameters()]))
        orig_unet_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.unet_adapter_modules.parameters()]))#加载UNET
  
        if pretrained_ip_adapter_path is not None:
          #  pretrained_ref_everything_state_dict = torch.load(pretrained_ref_everything_path, map_location="cpu")["module"]
            pretrained_ip_adapter_state_dict = torch.load(pretrained_ip_adapter_path, map_location="cpu")
            
            # Load state dict for image_proj_model and adapter_modules
            self.ipa_model.load_state_dict(pretrained_ip_adapter_state_dict["image_proj"], strict=True)
            self.unet_adapter_modules.load_state_dict(pretrained_ip_adapter_state_dict["ip_adapter"], strict=False)
            
            unet_dict = {}
            image_proj_dict = {}
            '''
            for k in pretrained_ref_everything_state_dict.keys():
                if k.startswith("unet"):
                    unet_dict[k.replace("unet.", "")] = pretrained_ref_everything_state_dict[k]
                elif k.startswith("image_proj"):
                    image_proj_dict[k.replace("image_proj.", "")] = pretrained_ref_everything_state_dict[k]
            '''
            for n, p in self.unet.named_parameters():
                if p.device.type == "meta":
                    print("META:", n)
            #self.ref_unet.load_state_dict(unet_dict, strict=False)
           # self.ref_ipa_model.load_state_dict(image_proj_dict, strict=False)
            
            # 初始化 unet.class_embedding
            # 策略：第一个投影层的权重随机初始化，其余（第一个偏置、第二个权重和偏置）都初始化为0
            if hasattr(self.unet, 'class_embedding') and self.unet.class_embedding is not None:
                print("初始化 class_embedding（第一个投影层权重随机，其余为0）...")
                # 收集所有线性层模块
                linear_modules = []
                for module_name, module in self.unet.class_embedding.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        linear_modules.append((module_name, module))
                
                # 遍历所有参数
                for module_name, module in self.unet.class_embedding.named_modules():
                    if module_name == '':  # 跳过根模块
                        continue
                    for param_name, param in list(module.named_parameters(recurse=False)):
                        # 判断是否是第一个投影层的权重
                        is_first_linear_weight = (
                            isinstance(module, torch.nn.Linear) and 
                            'weight' in param_name and
                            len(linear_modules) > 0 and
                            linear_modules[0][0] == module_name
                        )
                        
                        if param.device.type == "meta":
                            # 创建具化的参数（在CPU上）
                            if is_first_linear_weight:
                                # 第一个投影层的权重：使用 PyTorch nn.Linear 默认初始化
                                in_features = param.shape[1]  # 对于 Linear，weight 形状是 (out_features, in_features)
                                bound = 1.0 / math.sqrt(in_features)
                                materialized_param = torch.nn.Parameter(
                                    torch.empty(param.shape, dtype=torch.float32, device="cpu"),
                                    requires_grad=param.requires_grad
                                )
                                torch.nn.init.uniform_(materialized_param, -bound, bound)
                                print(f"  具化并初始化 (meta -> cpu, uniform): {module_name}.{param_name}, shape: {materialized_param.shape} [第一个投影层权重]")
                            else:
                                # 其余参数：初始化为0
                                materialized_param = torch.nn.Parameter(
                                    torch.zeros(param.shape, dtype=torch.float32, device="cpu"),
                                    requires_grad=param.requires_grad
                                )
                                print(f"  具化并初始化 (meta -> cpu, zeros): {module_name}.{param_name}, shape: {materialized_param.shape}")
                            # 直接替换模块中的参数
                            setattr(module, param_name, materialized_param)
                        else:
                            # 如果已经在真实设备上
                            if is_first_linear_weight:
                                # 第一个投影层的权重：使用 PyTorch nn.Linear 默认初始化
                                in_features = param.shape[1]
                                bound = 1.0 / math.sqrt(in_features)
                                torch.nn.init.uniform_(param, -bound, bound)
                                print(f"  初始化权重 (uniform): {module_name}.{param_name}, shape: {param.shape} [第一个投影层权重]")
                            else:
                                # 其余参数：初始化为0
                                torch.nn.init.zeros_(param)
                                print(f"  初始化 (zeros): {module_name}.{param_name}, shape: {param.shape}")
                print("class_embedding 初始化完成！")
            
            # Calculate new checksums
            new_ipa_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ipa_model.parameters()]))
           # new_unet_adapter1_sum = torch.sum(torch.stack([torch.sum(p) for p in self.unet_adapter_modules1.parameters()]))
            new_unet_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.unet_adapter_modules.parameters()]))
            #new_ref_unet_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ref_unet.parameters()]))
          #  new_ref_ipa_sum = torch.sum(torch.stack([torch.sum(p) for p in self.ref_ipa_model.parameters()]))
            
         
            # Verify if the weights have changed
            assert orig_unet_adapter_sum != new_unet_adapter_sum, "Weights of unet_adapter_modules did not change!"
            assert orig_ipa_sum != new_ipa_sum, "Weights of ipa_model did not change!"
           # assert orig_unet_adapter1_sum != new_unet_adapter1_sum, "Weights of unet_adapter_modules1 did not change!"
           # assert orig_ref_unet_sum != new_ref_unet_sum, "Weights of ref_unet did not change!"
          #  assert orig_ref_ipa_sum != new_ref_ipa_sum, "Weights of ref_ipa_model did not change!"

            print(f"Successfully loaded weights")
        else:
            print(f"No weights to load")
class ImageProjModel(torch.nn.Module):
    """Projection model - converts CLIP image features into a format suitable for UNet cross-attention"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim  # UNet cross-attention dimension 768 
        self.clip_extra_context_tokens = clip_extra_context_tokens  # Number of extra context tokens 4
        # Linear projection layer, converts CLIP embeddings into multiple extra context tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim) #1024 3072
        self.norm = torch.nn.LayerNorm(cross_attention_dim)  # Normalization layer 768 

    def forward(self, image_embeds):
        # Project CLIP image embeddings into multiple context tokens
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
        

def main():
  #  os.environ["NCCL_IB_DISABLE"] = "1"
  #  os.environ["NCCL_P2P_DISABLE"] = "1"
    # 强制使用 cuda:4
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    
    args = parse_args()
    print(args)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        project_dir=logging_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_stage3_name_or_path, subfolder="unet", class_embed_type="projection" ,projection_class_embeddings_input_dim=1024)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_stage3_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
   
    # Initialize IP-Adapter model components
    # Image projection model maps CLIP image embeddings to UNet's cross-attention dimension
    num_tokens = 4  # Number of extra context tokens
    ipa_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,#768
        clip_embeddings_dim=1024,#1024
        clip_extra_context_tokens=num_tokens,#4
    )
    ipa_weight = torch.load(args.pretrained_adapter_model_path, map_location="cpu")
    # load ipa weight
    ref_ipa_model = Resampler(
        dim=unet.config.cross_attention_dim, #768
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=1024,
        output_dim=unet.config.cross_attention_dim, #768
        ff_mult=4
    )
    # 从预训练权重中过滤掉 proj_in 相关参数（维度不匹配：1280->1024）
    pretrained_dict = ipa_weight['image_proj']
    filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('proj_in.')}
    
    # 记录被过滤的参数
    skipped_keys = [k for k in pretrained_dict.keys() if k.startswith('proj_in.')]
    print(f"跳过的 proj_in 参数: {skipped_keys}")
    print(f"成功加载 {len(filtered_dict)} 个参数，跳过 {len(skipped_keys)} 个参数")
    
    # 加载过滤后的权重（允许部分参数缺失）
    ref_ipa_model.load_state_dict(filtered_dict, strict=False)
    
    # 将 proj_in 层的权重和偏置初始化为零
    torch.nn.init.zeros_(ref_ipa_model.proj_in.weight)
    torch.nn.init.zeros_(ref_ipa_model.proj_in.bias)
    print(f"proj_in 层已零初始化: weight shape={ref_ipa_model.proj_in.weight.shape}, bias shape={ref_ipa_model.proj_in.bias.shape}")
    # set attention processor
    ref_attn_procs = {}
    st = ref_unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):#midblock只需要最后一层的输出结果
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):#从后到前面
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):#从前面到后面
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            ref_attn_procs[name] = SAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]
        else:
            ref_attn_procs[name] = RefCAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]

            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_p_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_p_ref.weight": st[layer_name + ".to_v.weight"],
            }
            ref_attn_procs[name].load_state_dict(weights)

    ref_unet.set_attn_processor(ref_attn_procs)
    del st
    
    # Initialize adapter attention processors
    # Create appropriate attention processors for each attention block in UNet
    # init adapter modules
    attn_procs = {}
    attn_procs_sa = {}  # Self-attention processors
    attn_procs_ca = {}  # Cross-attention processors
    unet_sd = unet.state_dict()
    # Initialize UNet's attention + IP-Adapter's cross_attention parameters
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            
        if cross_attention_dim is None:#SA 
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            attn_procs_sa[name] = attn_procs[name]  # Keep reference to SA processors
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ref.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ref.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:#CA 
            layer_name = name.split(".processor")[0]
            # Layers that need to add IP join additional attention parameters
            if 'down_blocks.0.attentions.1' in name:#第三组下采样阶段 第二个block 的CA
        
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }  
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=False)
                
                attn_procs[name].load_state_dict(weights, strict=False)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=True)#IPAttnProcessor2_0
            attn_procs_ca[name] = attn_procs[name]  # Keep reference to CA processors

    # Load all attention processing into UNet
    # Set all attention processors (both SA and CA)
    unet.set_attn_processor(attn_procs)
    # Extract SA and CA processors separately for later use
    unet_adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    del unet_sd
    
     # Create the complete IP-Adapter model
    ip_adapter = SD(
        unet,
        ref_unet,
        ref_ipa_model,
        ipa_model,
        unet_adapter_modules,
        args.pretrained_ip_adapter_path,
        inter_dim=2560,           # Use command-line arguments
        cross_heads=8,         # Use command-line arguments
        reshape_blocks=8,   # Use command-line arguments
        cross_value_dim=64, # Use command-line arguments
        fusion_method='cross_attention'
    )

   # ip_adapter.ref_unet.enable_gradient_checkpointing()
    
    # Enable gradient checkpointing to save memory
   # unet.enable_gradient_checkpointing()
    vae.requires_grad_(False)
    ip_adapter.ref_unet.requires_grad_(True)  
    ip_adapter.ref_ipa_model.requires_grad_(True)
    ip_adapter.unet.requires_grad_(False)
    ip_adapter.ipa_model.requires_grad_(True)
    ip_adapter.composed_modules.requires_grad_(True)
    ip_adapter.unet_adapter_modules.requires_grad_(True)
    ip_adapter.unet.class_embedding.requires_grad_(True)
    params_to_opt = itertools.chain(
        ip_adapter.ref_unet.parameters(),
        ip_adapter.ref_ipa_model.parameters(),
        ip_adapter.ipa_model.parameters(),  
        ip_adapter.composed_modules.parameters(),
        ip_adapter.unet_adapter_modules.parameters(),
        ip_adapter.unet.class_embedding.parameters()
    )
    accelerator.print("Trainable parameters: ref_unet:{:.2f}M, ref_ipa_model:{:.2f}M, ipa_model:{:.2f}M, composed_modules:{:.2f}M, unet_adapter_modules:{:.2f}M, unet.class_embedding:{:.2f}M".format(
        count_model_params(ip_adapter.ref_unet),
        count_model_params(ip_adapter.ref_ipa_model),
        count_model_params(ip_adapter.ipa_model),
        count_model_params(ip_adapter.composed_modules),
        count_model_params(ip_adapter.unet_adapter_modules),
        count_model_params(ip_adapter.unet.class_embedding)
       )
    )
    if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
        print(f"optimizer DummyOptim buzhixing: {optimizer}")
    

   # for name, param in unet.named_parameters():
    #    print(f"{name}: requires_grad={param.requires_grad}")
    for name, param in ip_adapter.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

        # TODO (patil-suraj): load scheduler using args
    noise_scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        rescale_betas_zero_snr=True,
        timestep_spacing="trailing", prediction_type="epsilon",
    )
    if args.image_root_path == "vitonhd":
        dataset = ImageDatasetvitonhd(
            image_root_path=args.image_root_path,
            json_file= args.json_path,
            size=(args.img_width, args.img_height),
            siglip_embeddings_root = args.siglip_embeddings_root,
            qwen_ipa_embeddings_root = args.qwen_ipa_embeddings_root,
            qwen_CA_embeddings_root = args.qwen_CA_embeddings_root,
            clip_embeddings_root = args.clip_embeddings_root,
            qwen_text_embeddings_root = args.qwen_text_embeddings_root,
        )  
    else:
        dataset = ImageDatasetDressCode(
            image_root_path=args.image_root_path,
            json_file= args.json_path,
            size=(args.img_width, args.img_height),
            siglip_embeddings_root = args.siglip_embeddings_root,
            qwen_ipa_embeddings_root = args.qwen_ipa_embeddings_root,
            qwen_CA_embeddings_root = args.qwen_CA_embeddings_root,
            clip_embeddings_root = args.clip_embeddings_root,
            qwen_text_embeddings_root = args.qwen_text_embeddings_root,
        ) 

    print("stage3_trainbufennewgengzhengwenben.py main_process_port 29501 ")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=4
    )

    if accelerator.state.deepspeed_plugin is not None:
        # here we use agrs.gradient_accumulation_steps
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
        print(f"lr_scheduler DummyScheduler buzhixing: ")
    

    if (
            accelerator.state.deepspeed_plugin is not None
            and accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == "auto"
    ):
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
   # ip_adapter.to(device="cuda",dtype=torch.float16)
    print("==== META CHECK ====")
    for n, p in ip_adapter.named_parameters():
        if p.device.type == "meta":
            print("NOMETA DATA")
    ip_adapter.to(device="cuda", dtype=torch.float16)
    ip_adapter, optimizer, lr_scheduler = accelerator.prepare(ip_adapter, optimizer, lr_scheduler)
    weight_dtype = torch.float16
    
    if accelerator.state.deepspeed_plugin is None:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    else:
        if accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]:
            weight_dtype = torch.bfloat16
    
    print(f"weight_dtype Final 🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉: {weight_dtype}")
    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
   
    vae.to(accelerator.device, dtype=weight_dtype)
  

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # New Code #
        # Loads the DeepSpeed checkpoint from the specified path
        last_epoch, last_global_step = load_training_checkpoint(
            ip_adapter,
            args.resume_from_checkpoint,
            **{"load_optimizer_states": True, "load_lr_scheduler_states": True},
        )
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        starting_epoch = last_epoch
        global_steps = last_global_step
    
    # 创建可视化输出目录
   # vis_output_dir = os.path.join(args.output_dir, "visualization")
   # os.makedirs(vis_output_dir, exist_ok=True)
    
    # 用于计算剩余时间的变量
    step_times = []
    max_step_times = 100  # 保留最近100个step的时间用于平均
    training_start_time = time.perf_counter()
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        train_loss = 0.0
        step = 0
        begin = time.perf_counter()
        # 用于累积32次迭代的时间和loss
        accumulated_step_time = 0.0
        accumulated_data_time = 0.0
        for batch in train_dataloader:
            # 可视化：每个epoch的第一个batch保存可视化图像

            load_data_time = time.perf_counter() - begin
            # Convert images to latent space
            with torch.no_grad():
                s_vae_img = vae.encode(
                    batch["s_vae_img"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                s_vae_img = s_vae_img * 0.18215

                im_mask_fine = vae.encode(
                    batch["im_mask_fine"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                im_mask_fine = im_mask_fine * 0.18215

                t_vae_img = vae.encode(
                    batch["t_vae_img"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                t_vae_img = t_vae_img * 0.18215

                warp_cloth = vae.encode(
                    batch["warp_cloth"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                warp_cloth = warp_cloth * 0.18215

            #prepare stage2 latents
           # stage2_channel1_combined_img = torch.cat([im_mask_fine, s_vae_img], dim=3)  #x axis concat
            mask_latent = torch.nn.functional.interpolate(batch["parse_cloth"].to(accelerator.device, dtype=weight_dtype), size=s_vae_img.shape[-2:], mode="nearest")
          #  stage2_channel2_mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=3)
         #   stage2_channel3_latents = torch.cat([s_vae_img, s_vae_img], dim=3)
            stage2_latents_concat = s_vae_img
            siglip_s_img = batch["siglip_s_img"].to(accelerator.device, dtype=weight_dtype)
            qwen_text_embeddings = batch["qwen_text_embeddings"].to(accelerator.device, dtype=weight_dtype)
            
            #prepare stage3 latents
         
            inshop_mask = batch["inshop_mask"].to(accelerator.device, dtype=weight_dtype)
            inshop_mask = torch.nn.functional.interpolate(inshop_mask, size=s_vae_img.shape[-2:], mode="nearest")
        
            stage3_channel3_latents = t_vae_img
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(stage3_channel3_latents)
            if args.noise_offset > 0:
                noise += args.noise_offset * torch.randn(
                    (stage3_channel3_latents.shape[0], stage3_channel3_latents.shape[1], 1, 1),
                    device=stage3_channel3_latents.device,
                )
            bsz = stage3_channel3_latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=stage3_channel3_latents.device)
            timesteps = timesteps.long()
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(stage3_channel3_latents, noise, timesteps)
    
            stage3_latents_concat_new = noisy_latents
            clip_t_img = batch["clip_t_img"].to(accelerator.device, dtype=weight_dtype)
            qwen_ipa_embeddings = batch["qwen_ipa_embeddings"].to(accelerator.device, dtype=weight_dtype)
            qwen_CA_embeddings = batch["qwen_CA_embeddings"].to(accelerator.device, dtype=weight_dtype)
            

            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(
                    stage3_channel3_latents, noise, timesteps
                )
            
            
           

            if args.dream_order == 'inf':
                # ========== 标准训练（无 DREAM）==========
                model_pred = ip_adapter(
                    stage2_latents_concat,
                    stage3_latents_concat_new,
                    qwen_text_embeddings,
                    siglip_s_img,
                    qwen_ipa_embeddings,
                    qwen_CA_embeddings,
                    clip_t_img,
                    timesteps,
                    inshop_mask,
                    mask_latent)
                target_final = target
            else:
                # ========== DREAM 训练（使用 Diffusers 官方实现）========== 
                dream_detail_preservation = float(args.dream_order)
                # 步骤1：冻结模型预测（不计算梯度）
                with torch.no_grad():
                    ip_adapter.eval()

                    frozen_pred = ip_adapter(
                    stage2_latents_concat,
                    stage3_latents_concat_new,
                    qwen_text_embeddings,
                    siglip_s_img,
                    qwen_ipa_embeddings,
                    qwen_CA_embeddings,
                    clip_t_img,
                    timesteps,
                    inshop_mask,
                    mask_latent)

                    ip_adapter.train()
                
                # 步骤2：计算预测误差（delta_noise）
                if noise_scheduler.config.prediction_type == "epsilon":
                    delta_noise = (target - frozen_pred).detach()
                else:
                    raise NotImplementedError("DREAM currently only supports epsilon prediction")

                # 步骤3：使用 Diffusers 官方公式计算校正
                # 参考：diffusers.training_utils.compute_dream_and_update_latents
                # 确保 alphas_cumprod 使用与模型一致的 dtype（fp16），避免 dtype 冲突
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=weight_dtype)[timesteps, None, None, None]
                sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
                
                # 论文公式：lambda = sqrt(1 - alpha) ** p，其中 p 是 dream_detail_preservation
                dream_lambda = sqrt_one_minus_alphas_cumprod ** dream_detail_preservation
                
                # 校正输入（noisy_latents）
                # 公式：noisy_latents' = noisy_latents + sqrt(1-α) * lambda * Δε
                noisy_latents_rectified = stage3_latents_concat_new + sqrt_one_minus_alphas_cumprod * dream_lambda * delta_noise
                
                # 校正目标（target）
                # 公式：target' = target + lambda * Δε
                target_final = target + dream_lambda * delta_noise

             

                model_pred = ip_adapter(
                    stage2_latents_concat,
                    noisy_latents_rectified,
                    qwen_text_embeddings,
                    siglip_s_img,
                    qwen_ipa_embeddings,
                    qwen_CA_embeddings,
                    clip_t_img,
                    timesteps,
                    inshop_mask,
                    mask_latent)

           
            if args.snr_gamma == 0:

                loss = F.mse_loss(
                    model_pred.float(), target_final.float(), reduction="mean"
                )
                
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                if noise_scheduler.config.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                )
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="none"
                )
                loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                )
                loss = loss.mean()
                

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item()

            # Backpropagate
            accelerator.backward(loss)

             # 累积时间和数据加载时间
            step_time = time.perf_counter() - begin
            accumulated_step_time += step_time
            accumulated_data_time += load_data_time
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # do nothing
                lr_scheduler.step()  # only for not deepspeed lr_scheduler
                optimizer.zero_grad()  # do nothing

                if accelerator.sync_gradients:
                  
                    accelerator.log({"train_loss": train_loss / args.gradient_accumulation_steps}, step=global_steps)
                    
                    # 额外打印32次累积的平均loss和时间
                    if accelerator.is_main_process:
                        print("执行avg_loss_accumulated")
                        avg_loss_accumulated = train_loss / args.gradient_accumulation_steps
                        avg_step_time_accumulated = accumulated_step_time / args.gradient_accumulation_steps
                        avg_data_time_accumulated = accumulated_data_time / args.gradient_accumulation_steps
                        logging.info(
                            "[累积1次] avg_loss: {:.4f}, avg_step_time: {:.2f}s, avg_data_time: {:.2f}s".format(
                                avg_loss_accumulated, avg_step_time_accumulated, avg_data_time_accumulated
                            )
                        )

                        

                        # 重置累积变量
                        accumulated_step_time = 0.0
                        accumulated_data_time = 0.0
                    
                    train_loss = 0.0

            if accelerator.is_main_process:
                # 计算剩余时间
                step_time = time.perf_counter() - begin
                step_times.append(step_time)
                if len(step_times) > max_step_times:
                    step_times.pop(0)
                
                avg_step_time = sum(step_times) / len(step_times)
                remaining_steps = args.max_train_steps - global_steps
                eta_seconds = avg_step_time * remaining_steps
                progress_percent = (global_steps / args.max_train_steps) * 100
                
                # 计算已用时间
                elapsed_time = time.perf_counter() - training_start_time
                
                logging.info(
                    "Epoch {}, step {}/{} ({:.1f}%), loss: {:.4f}, lr: {:.2e}, "
                    "step_time: {:.2f}s, data_time: {:.2f}s, 已用时间: {}, 预计剩余: {}".format(
                        epoch, global_steps, args.max_train_steps, progress_percent,
                        loss.detach().item(), lr_scheduler.get_lr()[0],
                        step_time, load_data_time,
                        format_time(elapsed_time), format_time(eta_seconds))
                )
            global_steps += 1
            step += 1

            # checkpoint
            if isinstance(checkpointing_steps, int):
                if global_steps % checkpointing_steps == 0:
                    checkpoint_model(args.output_dir, global_steps, ip_adapter, epoch, global_steps)

            # stop training
            if global_steps >= args.max_train_steps:
                break
            begin = time.perf_counter()

    accelerator.wait_for_everyone()
    # Save last model
    checkpoint_model(args.output_dir, global_steps, ip_adapter, epoch, global_steps)

    accelerator.end_training()


if __name__ == "__main__":
    main()

    '''
    if step == 0 and accelerator.is_main_process:
                # 获取第一张图的数据
                s_vae_img_vis = batch["s_vae_img"][0].cpu()  # (3, H, W), [-1, 1]
                im_mask_fine_vis = batch["im_mask_fine"][0].cpu()  # (3, H, W), [-1, 1]
                parse_cloth_vis = batch["parse_cloth"][0].cpu()  # (1, H, W), [0, 1]
                
                # 转换数值范围：[-1, 1] -> [0, 1]
                s_vae_img_vis = (s_vae_img_vis + 1) / 2
                im_mask_fine_vis = (im_mask_fine_vis + 1) / 2
                
                # 转换为numpy并调整维度顺序 (C, H, W) -> (H, W, C)
                s_vae_img_vis = s_vae_img_vis.permute(1, 2, 0).numpy()
                im_mask_fine_vis = im_mask_fine_vis.permute(1, 2, 0).numpy()
                parse_cloth_vis = parse_cloth_vis.squeeze(0).numpy()  # (H, W)
                
                # 创建可视化图像
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(s_vae_img_vis)
                axes[0].set_title('s_vae_img (原始图像)', fontsize=12)
                axes[0].axis('off')
                
                axes[1].imshow(im_mask_fine_vis)
                axes[1].set_title('im_mask_fine (遮罩图像)', fontsize=12)
                axes[1].axis('off')
                
                axes[2].imshow(parse_cloth_vis, cmap='gray')
                axes[2].set_title('parse_cloth (服装mask)', fontsize=12)
                axes[2].axis('off')
                
                plt.tight_layout()
                vis_path = os.path.join(vis_output_dir, f"epoch_{epoch}_step_{step}.png")
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                accelerator.print(f"可视化图像已保存到: {vis_path}")'''

            

    
