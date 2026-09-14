import os
import json
import torch
import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn
from safetensors.torch import load_file
from ip_adapter.attention_processornewsp3 import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
from adapter.resampler import Resampler
from adapter.attention_processor3two import SAttnProcessor2_0,RefCAttnProcessor2_0,RefSAttnProcessor2_0
from ip_adapter.attention_processor import Cross_Attention

class HarmonyAttention(nn.Module):
    def __init__(self,
                 image_hidden_size=1280,     # Input image feature dimension
                 text_context_dim=2048,      # Input text context feature dimension
                 inter_dim=2560,             # Intermediate projection dimension
                 cross_heads=10,             # Number of cross-attention heads
                 reshape_blocks=8,           # Number of image feature blocks
                 cross_value_dim=64,         # Value dimension per head after dimensionality reduction
                 scale=1.0,                  # Output scaling factor
                 fusion_method="qformer"):   # Fusion method selection: mlp, cross_attention, qformer
        super().__init__()
        
        self.scale = scale 
        self.reshape_blocks = reshape_blocks 
        self.cross_query_dim = inter_dim // reshape_blocks 
        self.fusion_method = fusion_method
        self.image_hidden_size = image_hidden_size
        self.text_context_dim = text_context_dim
        
        # Image projection required by all methods
        self.fc1 = nn.Linear(image_hidden_size, inter_dim)
        
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
        attended = self.fusion_text_image(x, text_embeds)  # [B, N_blocks, value_dim * heads]
        
        # Flatten, normalize, and project back
        attended = attended.view(B, -1)     # [B, flattened_dim]
        out = self.ln(attended)
        out = self.fc2(out) * self.scale

        return out

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

def prepare_extra_step_kwargs(generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}
        
        extra_step_kwargs["eta"] = eta

        
        extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt


def split_list_into_chunks(lst, n):
    """将列表分割成n个子列表，用于多GPU并行推理"""
    chunk_size = len(lst) // n
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    if len(chunks) > n:
        last_chunk = chunks.pop()
        chunks[-1].extend(last_chunk)
    return chunks


def prepare_models(args, device):
    """准备所有需要的模型，完全按照训练逻辑"""
    print(f"[GPU {device}] 正在加载模型...")
    
    # 加载基础模型
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path).to(dtype=torch.float16, device=device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_stage3_name_or_path, subfolder="unet", class_embed_type="projection", projection_class_embeddings_input_dim=1024)
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # 创建ref_ipa_model（Resampler，用于ref_unet分支）
    ref_ipa_model = Resampler(
        dim=unet.config.cross_attention_dim,  # 768
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=1024,  # SigLIP特征维度
        output_dim=unet.config.cross_attention_dim,  # 768
        ff_mult=4
    )

    # 创建ipa_model（ImageProjModel，用于主unet分支）
    num_tokens = 4  # Number of extra context tokens
    ipa_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,  # 768
        clip_embeddings_dim=1024,  # 1024
        clip_extra_context_tokens=num_tokens,  # 4
    )
    
    # 创建HarmonyAttention（composed_modules）
    composed_modules = HarmonyAttention(
        image_hidden_size=1024,      # Image feature dimension fixed
        text_context_dim=1024,        # Text context dimension fixed
        inter_dim=2560,
        cross_heads=8,
        reshape_blocks=8,
        cross_value_dim=64,
        scale=1.0,
        fusion_method='cross_attention'
    )
    
    # 设置ref_unet的attention processors
    ref_attn_procs = {}
    st = ref_unet.state_dict()
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
        
        if cross_attention_dim is None:
            ref_attn_procs[name] = SAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)
        else:
            ref_attn_procs[name] = RefCAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_p_ref.weight": st[layer_name + ".to_k.weight"],
                "to_v_p_ref.weight": st[layer_name + ".to_v.weight"],
            }
            ref_attn_procs[name].load_state_dict(weights)

    ref_unet.set_attn_processor(ref_attn_procs)
    del st
    
    # 设置主unet的attention processors
    attn_procs = {}
    unet_sd = unet.state_dict()
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
            
        if cross_attention_dim is None:  # SA 
            attn_procs[name] = RefSAttnProcessor2_0(name, hidden_size)
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ref.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ref.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name].load_state_dict(weights)
        else:  # CA 
            layer_name = name.split(".processor")[0]
            if 'down_blocks.0.attentions.1' in name:
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }  
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=False)
                attn_procs[name].load_state_dict(weights, strict=False)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=True)

    unet.set_attn_processor(attn_procs)
    unet_adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    del unet_sd
    
    # 加载训练好的权重
    print(f"[GPU {device}] 正在加载checkpoint: {args.model_stage3_ckpt}")
    model_sd = torch.load(args.model_stage3_ckpt, map_location="cpu")["module"]
    
    # 分类权重
    unet_dict = {}
    ref_unet_dict = {}
    ref_ipa_model_dict = {}
    ipa_model_dict = {}
    unet_adapter_modules_dict = {}
    composed_modules_dict = {}
    class_embeddings_dict = {}
    for k in model_sd.keys():
        if k.startswith("unet."):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        elif k.startswith("ref_unet."):
            ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
        elif k.startswith("ref_ipa_model."):
            ref_ipa_model_dict[k.replace("ref_ipa_model.", "")] = model_sd[k]
        elif k.startswith("ipa_model."):
            ipa_model_dict[k.replace("ipa_model.", "")] = model_sd[k]
        elif k.startswith("unet_adapter_modules."):
            unet_adapter_modules_dict[k.replace("unet_adapter_modules.", "")] = model_sd[k]
        elif k.startswith("composed_modules."):
            composed_modules_dict[k.replace("composed_modules.", "")] = model_sd[k]
        elif k.startswith("class_embedding."):
            class_embeddings_dict[k.replace("unet.class_embedding.", "")] = model_sd[k]
    
    # 在加载权重之前，先处理 meta 设备上的参数
    # 因为 meta 设备上的参数无法直接移动到 CPU，需要先替换它们
    print(f"[GPU {device}] 检查并处理 meta 设备上的参数...")
    meta_params_found = False
    for name, param in unet.named_parameters():
        if hasattr(param, 'device') and param.device.type == 'meta':
            meta_params_found = True
            # 获取参数路径
            parts = name.split('.')
            parent = unet
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            
            # 如果 checkpoint 中有对应的权重，使用 checkpoint 的权重创建新参数
            if name in unet_dict:
                loaded_weight = unet_dict[name].clone().to('cpu')
                new_param = torch.nn.Parameter(loaded_weight)
                
                # 检查是否全0
                is_all_zero = (loaded_weight.abs() < 1e-6).all().item()
                zero_ratio = (loaded_weight.abs() < 1e-6).float().mean().item()
                
                if is_all_zero:
                    print(f"[GPU {device}] ⚠️  警告: 替换 meta 参数 {name}，权重全为0！")
                elif zero_ratio > 0.9:
                    print(f"[GPU {device}] ⚠️  警告: 替换 meta 参数 {name}，{zero_ratio*100:.1f}%的权重为0")
                else:
                    print(f"[GPU {device}] 替换 meta 参数 {name}，使用 checkpoint 中的权重 (非零比例: {(1-zero_ratio)*100:.1f}%)")
                
                # 如果是class_embedding相关的权重，特别标注
                if 'class_embedding' in name:
                    weight_stats = f"mean={loaded_weight.mean().item():.6f}, std={loaded_weight.std().item():.6f}, min={loaded_weight.min().item():.6f}, max={loaded_weight.max().item():.6f}"
                    print(f"[GPU {device}]   class_embedding权重统计: {weight_stats}")
                
            else:
                # 如果没有对应的权重，根据参数形状创建一个零参数
                # meta 设备上的参数仍然有 shape 属性，可以使用它
                try:
                    param_shape = param.shape
                    new_param = torch.nn.Parameter(torch.zeros(param_shape, device='cpu'))
                    print(f"[GPU {device}] 替换 meta 参数 {name}，使用零初始化（形状: {param_shape}）")
                except Exception as e:
                    print(f"[GPU {device}] 警告: 无法处理参数 {name}，错误: {e}，跳过")
                    continue
            
            # 替换参数
            setattr(parent, attr_name, new_param)
    
    if meta_params_found:
        print(f"[GPU {device}] 已处理所有 meta 设备上的参数")
    
    # 现在可以安全地将 unet 移动到 CPU
    unet = unet.to("cpu")
    
    # 加载权重到各个模块
    print(f"[GPU {device}] 加载 unet 权重 ({len(unet_dict)} 个参数)...")
    # 使用 assign=True 来强制赋值，避免 meta 设备的问题（PyTorch 2.0+）
    # 如果 PyTorch 版本不支持 assign 参数，会抛出 TypeError，则使用普通方式
    try:
        unet.load_state_dict(unet_dict, strict=False, assign=True)
    except TypeError:
        # 对于不支持 assign 的版本，使用普通方式加载
        unet.load_state_dict(unet_dict, strict=False)
    
    print(f"[GPU {device}] 加载 ref_unet 权重 ({len(ref_unet_dict)} 个参数)...")
    ref_unet.load_state_dict(ref_unet_dict, strict=False)
    
    print(f"[GPU {device}] 加载 ref_ipa_model 权重 ({len(ref_ipa_model_dict)} 个参数)...")
    ref_ipa_model.load_state_dict(ref_ipa_model_dict, strict=False)
    
    print(f"[GPU {device}] 加载 ipa_model 权重 ({len(ipa_model_dict)} 个参数)...")
    ipa_model.load_state_dict(ipa_model_dict, strict=False)
    
    print(f"[GPU {device}] 加载 unet_adapter_modules 权重 ({len(unet_adapter_modules_dict)} 个参数)...")
    unet_adapter_modules.load_state_dict(unet_adapter_modules_dict, strict=False)
    
    print(f"[GPU {device}] 加载 composed_modules 权重 ({len(composed_modules_dict)} 个参数)...")
    composed_modules.load_state_dict(composed_modules_dict, strict=False)
    
    if len(class_embeddings_dict) > 0:
        print(f"[GPU {device}] 加载 class_embedding 权重 ({len(class_embeddings_dict)} 个参数)...")
        
        # 检查class_embedding权重是否全0
        for param_name, param_value in class_embeddings_dict.items():
            is_all_zero = (param_value.abs() < 1e-6).all().item()
            zero_ratio = (param_value.abs() < 1e-6).float().mean().item()
            if is_all_zero:
                print(f"[GPU {device}] ⚠️  警告: class_embedding.{param_name} 权重全为0！")
            elif zero_ratio > 0.9:
                print(f"[GPU {device}] ⚠️  警告: class_embedding.{param_name} 有{zero_ratio*100:.1f}%的权重为0")
            else:
                weight_stats = f"mean={param_value.mean().item():.6f}, std={param_value.std().item():.6f}, min={param_value.min().item():.6f}, max={param_value.max().item():.6f}"
                print(f"[GPU {device}]   class_embedding.{param_name}: {weight_stats} (非零比例: {(1-zero_ratio)*100:.1f}%)")
        
        if hasattr(unet, 'class_embedding') and unet.class_embedding is not None:
            unet.class_embedding.load_state_dict(class_embeddings_dict, strict=False)
        else:
            print(f"[GPU {device}] 警告: unet 没有 class_embedding 属性，跳过加载")
    
    print(f"[GPU {device}] 所有权重加载完成！")
    
    # 移动到GPU并设置数据类型
    unet = unet.to(dtype=torch.float16, device=device)
    ref_unet = ref_unet.to(dtype=torch.float16, device=device)
    ref_ipa_model = ref_ipa_model.to(dtype=torch.float16, device=device)
    ipa_model = ipa_model.to(dtype=torch.float16, device=device)
    unet_adapter_modules = unet_adapter_modules.to(dtype=torch.float16, device=device)
    composed_modules = composed_modules.to(dtype=torch.float16, device=device)
    
    # 创建scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # 设置为eval模式
    vae.eval()
    unet.eval()
    ref_unet.eval()
    ref_ipa_model.eval()
    ipa_model.eval()
    composed_modules.eval()
    
    unet_adapter_modules.eval()
    print(f"[GPU {device}] 模型加载完成！")
    
    return {
        'vae': vae,
        'unet': unet,
        'ref_unet': ref_unet,
        'ref_ipa_model': ref_ipa_model,
        'ipa_model': ipa_model,
        'unet_adapter_modules': unet_adapter_modules,
        'composed_modules': composed_modules,
        'scheduler': noise_scheduler
    }


def prepare_data(item, args, device):
    """准备单个样本的所有输入数据（支持 VitonHD 和 DressCode）"""
    
    # 图像转换
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # 判断数据集类型
    is_dresscode = "DressCode" in args.image_root_path
    
    # 1. 加载source image
    # 使用 BICUBIC 插值进行下采样，保持图像质量
    s_img_path = os.path.join(args.image_root_path, item["source_image"])
    s_vae_img = Image.open(s_img_path).convert("RGB").resize(
        (args.img_width, args.img_height), Image.BICUBIC)
    s_vae_img_tensor = img_transform(s_vae_img).unsqueeze(0)  # [1, 3, H, W]
    
    # 2. 加载image-parse生成parse_cloth mask
    im_name = os.path.basename(item["source_image"])
    split = item["source_image"].split('/')[0]  # VitonHD: "train"/"test", DressCode: "upper_body"/"lower_body"/"dresses"
    
    if is_dresscode:
        # DressCode: 文件名 000000_0.jpg -> 000000_4.png
        parse_name = im_name.replace('_0.jpg', '_4.png')
        parse_path = os.path.join(args.image_root_path, split, 'label_maps', parse_name)
        category = split  # 保存类别信息
    else:
        # VitonHD: 直接替换扩展名
        parse_name = im_name.replace('.jpg', '.png')
        parse_path = os.path.join(args.image_root_path, split, 'image-parse-v3', parse_name)
    
    im_parse = Image.open(parse_path).resize((args.img_width, args.img_height), Image.NEAREST)
    parse_array = np.array(im_parse)
    
    # 根据数据集类型生成parse_cloth mask
    if is_dresscode:
        # DressCode: 根据类别确定标签
        if category == 'dresses':
            parse_cloth = (parse_array == 7).astype(np.float32)
        elif category == 'upper_body':
            parse_cloth = (parse_array == 4).astype(np.float32)
        elif category == 'lower_body':
            parse_cloth = (parse_array == 6).astype(np.float32)
        else:
            raise ValueError(f"未知的 DressCode 类别: {category}")
    else:
            # VitonHD: 类别5, 6, 7为衣服
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                    (parse_array == 6).astype(np.float32) + \
                    (parse_array == 7).astype(np.float32)
    
    parse_cloth = torch.from_numpy(parse_cloth).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 3. 计算im_mask_fine = s_vae_img * (1 - parse_cloth)
    im_mask_fine = s_vae_img_tensor * (1 - parse_cloth)
    
    # 4. 加载siglip embeddings（用于ref_unet）
    filename = os.path.basename(item["source_image"])
    stem = Path(filename).stem
    
    if is_dresscode:
        # DressCode: 使用 "test" 目录
        feature_path = os.path.join(args.siglip_embeddings_root, "test", f"{stem}.safetensors")
    else:
        # VitonHD: 使用 split ("train"/"test")
        feature_path = os.path.join(args.siglip_embeddings_root, split, f"{stem}.safetensors")
    
    siglip_s_img = load_file(feature_path)["embedding"].unsqueeze(0)  # [1, seq_len, 1024]
    
    # 5. 加载qwen text embeddings（用于ref_unet）
    if is_dresscode:
        qwen_text_path = os.path.join(args.qwen_text_embeddings_root, "test", f"{stem}.safetensors")
    else:
        qwen_text_path = os.path.join(args.qwen_text_embeddings_root, split, f"{stem}.safetensors")
    qwen_text_embeddings = load_file(qwen_text_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 5.1 加载empty qwen text embeddings（用于ref_unet的CFG无条件分支）
    empty_text_path = os.path.join(args.empty_qwen_text_embeddings_root)
    empty_qwen_text_embeddings = load_file(empty_text_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 6. 加载qwen_ipa embeddings（用于composed_modules）
    if is_dresscode:
        qwen_ipa_path = os.path.join(args.qwen_ipa_embeddings_root, "test", f"{stem}.safetensors")
    else:
        qwen_ipa_path = os.path.join(args.qwen_ipa_embeddings_root, split, f"{stem}.safetensors")
    qwen_ipa_embeddings = load_file(qwen_ipa_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 6.1 加载empty qwen_ipa embeddings（用于主unet的CFG）
    empty_qwen_ipa_path = os.path.join(args.empty_qwen_ipa_embeddings_root)
    empty_qwen_ipa_embeddings = load_file(empty_qwen_ipa_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 7. 加载qwen_CA embeddings（用于主unet的cross-attention）
    if is_dresscode:
        qwen_CA_path = os.path.join(args.qwen_CA_embeddings_root, "test", f"{stem}.safetensors")
    else:
        qwen_CA_path = os.path.join(args.qwen_CA_embeddings_root, split, f"{stem}.safetensors")
    qwen_CA_embeddings = load_file(qwen_CA_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 7.1 加载empty qwen_CA embeddings（用于主unet的CFG）
    empty_qwen_CA_path = os.path.join(args.empty_qwen_CA_embeddings_root)
    empty_qwen_CA_embeddings = load_file(empty_qwen_CA_path)["embedding"].unsqueeze(0)  # [1, seq_len, 768]
    
    # 8. 加载clip embeddings（用于主unet的class_labels和ipa_model）
    if is_dresscode:
        stemdress_code = stem.replace("_0", "_1")
        stage1_clip_t_path = os.path.join(args.stage1_clip_embeddings_root, "test", f"{stemdress_code}.safetensors")
    else:
        stage1_clip_t_path = os.path.join(args.stage1_clip_embeddings_root, split, f"{stem}.safetensors")
    stage1_clip_t_img = load_file(stage1_clip_t_path)["embedding"].unsqueeze(0)  # [1, 1024]
    
    # 8.1 加载zero clip embeddings（用于主unet的CFG）
    zero_clip_path = os.path.join(args.zero_clip_embeddings_root)
    zero_clip_img = load_file(zero_clip_path)["embedding"].unsqueeze(0)  # [1, 1024]
    zero_clip_img = torch.zeros_like(zero_clip_img)  # 直接设置为全0
    
   
   
    
    return {
        's_vae_img': s_vae_img_tensor.to(device),          # [1, 3, H, W]
        'im_mask_fine': im_mask_fine.to(device),           # [1, 3, H, W]
        'parse_cloth': parse_cloth.to(device),             # [1, 1, H, W]      # [1, 3, H, W]           # [1, 1, H, W]
        'siglip_s_img': siglip_s_img.to(device),           # [1, seq_len, 1024]
        'qwen_text_embeddings': qwen_text_embeddings.to(device),  # [1, seq_len, 768]
        'empty_qwen_text_embeddings': empty_qwen_text_embeddings.to(device),  # [1, seq_len, 768]
        'qwen_ipa_embeddings': qwen_ipa_embeddings.to(device),    # [1, seq_len, 768]
        'empty_qwen_ipa_embeddings': empty_qwen_ipa_embeddings.to(device),  # [1, seq_len, 768]
        'qwen_CA_embeddings': qwen_CA_embeddings.to(device),      # [1, seq_len, 768]
        'empty_qwen_CA_embeddings': empty_qwen_CA_embeddings.to(device),  # [1, seq_len, 768]
        'stage1_clip_t_img': stage1_clip_t_img.to(device),               # [1, 1024]
        'zero_clip_img': zero_clip_img.to(device),                       # [1, 1024]
        'original_image': s_vae_img                        # PIL Image for saving
    }


@torch.no_grad()
def inference_single_sample(data, models, args, generator):
    """对单个样本进行DDIM推理（与训练时的forward逻辑一致）"""
    eta = 0.0
    vae = models['vae']
    unet = models['unet']
    ref_unet = models['ref_unet']
    ref_ipa_model = models['ref_ipa_model']
    ipa_model = models['ipa_model']
    composed_modules = models['composed_modules']
    scheduler = models['scheduler']
    device = next(unet.parameters()).device
    weight_dtype = torch.float16
    
    # 1. 编码图像到latent空间
    s_vae_img_latent = vae.encode(data['s_vae_img'].to(dtype=weight_dtype)).latent_dist.mean
    s_vae_img_latent = s_vae_img_latent * 0.18215
    
    im_mask_fine_latent = vae.encode(data['im_mask_fine'].to(dtype=weight_dtype)).latent_dist.mean
    im_mask_fine_latent = im_mask_fine_latent * 0.18215
    

    
    
    # 2. 准备stage2的输入（用于ref_unet）
    stage2_channel1_combined_img = torch.cat([im_mask_fine_latent, s_vae_img_latent], dim=3)  # x axis concat
    mask_latent = torch.nn.functional.interpolate(
        data['parse_cloth'].to(dtype=weight_dtype), 
        size=s_vae_img_latent.shape[-2:], 
        mode="nearest"
    )
    stage2_channel2_mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=3)
    stage2_channel3_latents = s_vae_img_latent
    stage2_latents_concat = stage2_channel3_latents
    
    # 3. 准备stage3的条件输入
   
   
    # 4. 处理文本和图像embeddings
    qwen_text_embeddings = data['qwen_text_embeddings'].to(dtype=weight_dtype)
    empty_qwen_text_embeddings = data['empty_qwen_text_embeddings'].to(dtype=weight_dtype)
    siglip_s_img = data['siglip_s_img'].to(dtype=weight_dtype)
    qwen_ipa_embeddings = data['qwen_ipa_embeddings'].to(dtype=weight_dtype)
    empty_qwen_ipa_embeddings = data['empty_qwen_ipa_embeddings'].to(dtype=weight_dtype)
    qwen_CA_embeddings = data['qwen_CA_embeddings'].to(dtype=weight_dtype)
    empty_qwen_CA_embeddings = data['empty_qwen_CA_embeddings'].to(dtype=weight_dtype)
    stage1_clip_t_img = data['stage1_clip_t_img'].to(dtype=weight_dtype)
    zero_clip_img = data['zero_clip_img'].to(dtype=weight_dtype)
    
    # 5. 运行ref_unet一次，batch=1正常前向传播（只运行条件分支）
    ref_timesteps = torch.zeros((1,), device=device, dtype=torch.long)  # batch=1
    
    # 处理siglip特征（只使用条件分支）
    siglip_s_img_proj = ref_ipa_model(siglip_s_img)  # [1, 16, 768]
    _ = ref_unet(
        stage2_latents_concat,
        ref_timesteps,
        encoder_hidden_states=None,
        return_dict=False,
        cross_attention_kwargs={
            "image_hidden_states": siglip_s_img_proj,
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
    # 6. 获取self-attention缓存（ref_unet只运行了batch=1，直接使用缓存）
    for name in ref_unet.attn_processors.keys():
        if name.endswith("attn1.processor"):
            cached_hidden_states = ref_unet.attn_processors[name].cache["hidden_states"]  # [1, N, C]
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
        
            W_single_scaled = W_single // scale
            target_size = (H, W_single_scaled)  # 高度和宽度都按scale缩放
            
            if mask_latent is not None:
                mask_latent_scaled = torch.nn.functional.interpolate(
                    mask_latent, size=target_size, mode="nearest"
                )
                # 展平为 (B, H*W) 形式
                B_mask, C_mask, H_mask, W_mask = mask_latent_scaled.shape
                mask_latent_flat = mask_latent_scaled.view(B_mask, C_mask, H_mask * W_mask).squeeze(1)  # (B, H*W)
                mask_latent_dict[name] = mask_latent_flat
                mask_latent_expanded = mask_latent_flat.unsqueeze(-1)  # (B, H*W, 1)
                   # 衣服区域（mask=1）保留原始值，其他区域（mask=0）变成0
                cached_hidden_states = cached_hidden_states * mask_latent_expanded  # (B, H*W, C)
               
            sa_hidden_states[name] = cached_hidden_states
    
    # 7. 处理clip特征用于主unet（条件和无条件）
    # 条件分支qwen_CA_embeddings
    composed_embeds_cond = composed_modules(qwen_ipa_embeddings, stage1_clip_t_img)
    clip_t_img_enhanced_cond = stage1_clip_t_img + composed_embeds_cond
    ip_tokens_cond = ipa_model(clip_t_img_enhanced_cond)
    encoder_hidden_states_cond = torch.cat([qwen_CA_embeddings, ip_tokens_cond], dim=1)
    
    # 无条件分支
    composed_embeds_uncond = composed_modules(empty_qwen_ipa_embeddings, zero_clip_img)
    clip_t_img_enhanced_uncond = zero_clip_img + composed_embeds_uncond
    ip_tokens_uncond = ipa_model(torch.zeros_like(clip_t_img_enhanced_uncond))
    encoder_hidden_states_uncond = torch.cat([empty_qwen_CA_embeddings, ip_tokens_uncond], dim=1)
    
    # 8. 设置scheduler
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    
    # 9. 初始化latents（从随机噪声开始）
    latents_shape = (1, 4, s_vae_img_latent.shape[2], s_vae_img_latent.shape[3])  # [1, 4, H//8, 2*W//8]
    
    # 10. 初始化随机噪声
    noise = randn_tensor(latents_shape, generator=generator, device=device, dtype=weight_dtype)
    latents = noise * scheduler.init_noise_sigma
    
    # Prepare extra step kwargs
    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta)
    
    # 11. DDIM去噪循环（条件和无条件分开推理）
    for i, t in enumerate(tqdm(timesteps, desc="DDIM 去噪步骤", leave=False)):
        t_scalar = t.unsqueeze(0)  # [1]#stage1_clip_t_img
        
        # 条件分支推理（使用sa_hidden_states）
        noise_pred_cond = unet(
            latents,  # [1, 4, H//8, W//8]
            t_scalar,
            class_labels=stage1_clip_t_img,  # [1, 1024]
            encoder_hidden_states=encoder_hidden_states_cond,  # [1, seq_len, 768]
            cross_attention_kwargs={
                "sa_hidden_states": sa_hidden_states,  # 只传递给条件分支
                "inshop_mask": None,
                "mask_latent": None,
                "train": False
            }
        ).sample
        
        # 无条件分支推理（不使用sa_hidden_states）
        noise_pred_uncond = unet(
            latents,  # [1, 4, H//8, W//8]
            t_scalar,
            class_labels=zero_clip_img,  # [1, 1024]
            encoder_hidden_states=encoder_hidden_states_uncond,  # [1, seq_len, 768]
            cross_attention_kwargs={
                "sa_hidden_states": None,  # 无条件分支不使用sa_hidden_states
                "inshop_mask": None,
                "mask_latent": None,
                "train": False
            }
        ).sample
        
        # 应用CFG
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 去噪一步
        latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    

 
    latents = latents / 0.18215
    image = vae.decode(latents.to(dtype=weight_dtype)).sample
    
    # 13. 后处理：[-1, 1] -> [0, 255]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    
    return Image.fromarray(image)


def main(args, rank, test_data_subset):
    """主推理函数"""
    device = torch.device(f"cuda:{rank}")
    
    # 创建保存目录 - 根据数据集类型自动设置
    if "DressCode" in args.image_root_path:
        # DressCode: 使用指定的 output_path
        save_dir = os.path.join(args.output_path, f"seed{args.seed}_steps{args.num_inference_steps}_cfg{args.guidance_scale}")
    else:
        # VitonHD: 使用指定的 output_path
        save_dir = os.path.join(args.output_path, f"seed{args.seed}_steps{args.num_inference_steps}_cfg{args.guidance_scale}")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"[GPU {rank}] 保存路径: {save_dir}")
    
    # 加载模型
    models = prepare_models(args, device)
    
    # 设置随机种子
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    
    # 推理循环
    start_time = time.time()
    for idx, item in enumerate(test_data_subset):
        try:
            # 准备数据
            data = prepare_data(item, args, device)
            
            # 推理
            output_image = inference_single_sample(data, models, args, generator)
            
            # 保存结果
            source_filename = os.path.basename(item["target_image"])
            
            # 分别保存原图和生成图
            original_img = data['original_image'].resize((output_image.width, output_image.height))
            
            # 构建文件名
            filename_stem = Path(source_filename).stem
            # 如果是DressCode数据集，使用 basename_label 的形式保存
            if "DressCode" in args.image_root_path:
                original_save_path = os.path.join(save_dir, f"{filename_stem}_{item['label']}_original.png")
                generated_save_path = os.path.join(save_dir, f"{filename_stem}_{item['label']}.png")
            else:
                original_save_path = os.path.join(save_dir, f"{filename_stem}_original.png")
                generated_save_path = os.path.join(save_dir, f"{filename_stem}.png")
            
            # 分别保存
          #  original_img.save(original_save_path)
            output_image.save(generated_save_path)
            
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(test_data_subset) - idx - 1)
                print(f"[GPU {rank}] 进度: {idx+1}/{len(test_data_subset)}, "
                      f"平均耗时: {avg_time:.2f}s/张, 预计剩余: {remaining/60:.1f}分钟")
        
        except Exception as e:
            print(f"[GPU {rank}] 处理 {item['source_image']} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    print(f"[GPU {rank}] 完成！共处理 {len(test_data_subset)} 张图像，总耗时: {total_time/60:.2f}分钟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage2 Test Script with Multi-GPU Support')
    
    # 模型路径
    parser.add_argument('--model_stage3_ckpt', type=str, 
                        default="",
                        help='stage2Model')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default="SG161222/Realistic_Vision_V4.0_noVAE",
                        help='预训练模型路径')
    parser.add_argument('--pretrained_model_stage3_name_or_path', type=str,
                        default="SG161222/Realistic_Vision_V4.0_noVAE",
                        help='预训练模型路径')
    parser.add_argument('--pretrained_vae_model_path', type=str,
                        default="stabilityai/sd-vae-ft-mse",
                        help='VAE模型路径')
    
    # #DressCode/dresscode_test.json  vitonhd/vitonhd_test.json
    parser.add_argument('--json_path', type=str, 
                        default="DressCode/dresscode_test.json",
                        help='测试集JSON文件路径')
    parser.add_argument('--image_root_path', type=str, 
                        default="DressCode",
                        help='图像根目录')
    parser.add_argument(
        "--qwen_text_embeddings_root",
        type=str,
        default="DressCode/qwen_text_embeddings",
        help="Path to qwen text embeddings file.",
    )
    parser.add_argument(
        "--empty_qwen_text_embeddings_root",
        type=str,
        default="DressCode/qwen_text_embeddings/empty_text_embedding.safetensors",
        help="Path to empty qwen text embeddings file for CFG.",
    )
    parser.add_argument(
        "--qwen_ipa_embeddings_root",
        type=str,
        default="DressCode/qwen_ipa_text_embeddings",
        help="Path to qwen ipa embeddings file.",
    )
    parser.add_argument(
        "--empty_qwen_ipa_embeddings_root",
        type=str,
        default="DressCode/qwen_ipa_text_embeddings/empty_text_embedding.safetensors",
        help="Path to empty qwen ipa embeddings file for CFG.",
    )
    parser.add_argument(
        "--qwen_CA_embeddings_root",
        type=str,
        default="DressCode/qwen_ca_text_embeddings",
        help="Path to qwen CA embeddings file.",
    )
    parser.add_argument(
        "--empty_qwen_CA_embeddings_root",
        type=str,
        default="DressCode/qwen_ca_text_embeddings/empty_text_embedding.safetensors",
        help="Path to empty qwen CA embeddings file for CFG.",
    )
    parser.add_argument(
        "--stage1_clip_embeddings_root",
        type=str,
        default="DressCode/clip_embeddings",
        help="Path to clip image embeddings file.",
    )
    parser.add_argument(
        "--zero_clip_embeddings_root",
        type=str,
        default="DressCode/clip_embeddings/zero_image.safetensors",
        help="Path to zero clip embeddings file for CFG.",
    )
    parser.add_argument('--siglip_embeddings_root', type=str,
                        default="DressCode/siglip_embeddings",
                        help='SigLIP特征根目录')
    parser.add_argument(
        "--empty_siglip_embeddings_root",
        type=str,
        default="DressCode/siglip_embeddings/zero_image.safetensors",
        help="Path to empty siglip embeddings file for CFG.",
    )

    
    # 输出路径
    parser.add_argument('--output_path', type=str, 
                        default="",
                        help='OutPutPath')
    
    # 推理参数
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='DDIM推理步数')
    parser.add_argument('--guidance_scale', type=float, default=3,
                        help='Classifier-free guidance系数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # 图像尺寸
    parser.add_argument('--img_width', type=int, default=384,
                        help='图像宽度')
    parser.add_argument('--img_height', type=int, default=512,
                        help='图像高度')
    
    args = parser.parse_args()

    # 检测数据集类型
    is_dresscode = "DressCode" in args.image_root_path
    dataset_type = "DressCode" if is_dresscode else "VitonHD"
    
    print("="*60)
    print(f"Stage3 测试配置 ({dataset_type}):")
    print(f"  模型权重: {args.model_stage3_ckpt}")
    print(f"  预训练模型路径: {args.pretrained_model_name_or_path}")
    print(f"  预训练模型路径: {args.pretrained_model_stage3_name_or_path}")
    print(f"  VAE模型路径: {args.pretrained_vae_model_path}")
    print(f"  测试集JSON文件路径: {args.json_path}")
    print(f"  图像根目录: {args.image_root_path}")
    print(f"  Qwen文本描述文件路径: {args.qwen_text_embeddings_root}")
    print(f"  Qwen IPA文本描述文件路径: {args.qwen_ipa_embeddings_root}")
    print(f"  Qwen CA文本描述文件路径: {args.qwen_CA_embeddings_root}")
    print(f"  Clip图像特征根目录: {args.stage1_clip_embeddings_root}")
    print(f"  SigLIP特征根目录: {args.siglip_embeddings_root}")
    print(f"  图像尺寸: {args.img_width}x{args.img_height}")
    print(f"  输出目录: {args.output_path}")
    print(f"  推理步数: {args.num_inference_steps}")
    print(f"  Guidance Scale: {args.guidance_scale}")
    print("="*60)
    
    # 加载测试集
    with open(args.json_path, 'r') as f:
        test_data = json.load(f)
    print(f"测试集大小: {len(test_data)} 张图像")
    
    # 检测可用GPU数量
    num_gpus = 4
    print(f"检测到 {num_gpus} 个GPU")
    
    if num_gpus > 1:
        # 多GPU并行推理
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        
        # 分割数据
        data_chunks = split_list_into_chunks(test_data, num_gpus)
        
        # 创建进程
        processes = []
        for rank in range(num_gpus):
            p = mp.Process(target=main, args=(args, rank, data_chunks[rank]))
            processes.append(p)
            p.start()
        
        # 等待所有进程完成
        for p in processes:
            p.join()
    else:
        # 单GPU推理
        main(args, 0, test_data)
    
    print("="*60)
    print("测试完成！")
    print("="*60)
