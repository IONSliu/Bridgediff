import os
import json
import random
import torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from diffusers.image_processor import VaeImageProcessor
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import numpy as np




__all__ = ['PriorImageDataset', 'PriorImageVitonHDDataset', 'PriorCollate_fn', 
           'read_coordinates_file', 'read_openpose_keypoints']




def collate_fn(data):
   # clip_s_img = torch.stack([example["clip_s_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
   # clip_t_img = torch.stack([example["clip_t_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
    siglip_s_img = torch.stack([example["siglip_s_img"] for example in data]).to(memory_format=torch.contiguous_format)
    im_mask_fine = torch.stack([example["im_mask_fine"] for example in data]).to(memory_format=torch.contiguous_format).float()
    parse_cloth = torch.stack([example["parse_cloth"] for example in data]).to(memory_format=torch.contiguous_format).float()
    s_vae_img = torch.stack([example["s_vae_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
    warp_cloth = torch.stack([example["warp_cloth"] for example in data]).to(memory_format=torch.contiguous_format).float()
    qwen_ipa_embeddings = torch.stack([example["qwen_ipa_embeddings"] for example in data]).to(memory_format=torch.contiguous_format)
    qwen_CA_embeddings = torch.stack([example["qwen_CA_embeddings"] for example in data]).to(memory_format=torch.contiguous_format)
    clip_t_img = torch.stack([example["clip_t_img"] for example in data]).to(memory_format=torch.contiguous_format)
    empty_qwen_text_embeddings = torch.stack([example["empty_qwen_text_embeddings"] for example in data]).to(memory_format=torch.contiguous_format)
    qwen_text_embeddings = torch.stack([example["qwen_text_embeddings"] for example in data]).to(memory_format=torch.contiguous_format)
    t_vae_img = torch.stack([example["t_vae_img"] for example in data]).to(memory_format=torch.contiguous_format).float()
    inshop_mask = torch.stack([example["inshop_mask"] for example in data]).to(memory_format=torch.contiguous_format).float()
    return {


        "siglip_s_img": siglip_s_img,
        "im_mask_fine": im_mask_fine,
        "parse_cloth": parse_cloth,
        "s_vae_img": s_vae_img,
        "warp_cloth": warp_cloth,
        "qwen_ipa_embeddings": qwen_ipa_embeddings,
        "qwen_CA_embeddings": qwen_CA_embeddings,
        "clip_t_img": clip_t_img,
        "empty_qwen_text_embeddings": empty_qwen_text_embeddings,
        "qwen_text_embeddings": qwen_text_embeddings,
        "t_vae_img": t_vae_img,
        "inshop_mask": inshop_mask

    }


class ImageDatasetvitonhd(Dataset):
    def __init__(
        self,

        image_root_path,
        json_file,
        size=(512,512),
        siglip_embeddings_root=None,
        qwen_ipa_embeddings_root=None,
        qwen_CA_embeddings_root=None,
        clip_embeddings_root=None,
        qwen_text_embeddings_root=None
        
    ):
        self.data = json.load(open(json_file))
        self.image_root_path = image_root_path

        self.size = size
        self.im_names = [os.path.basename(item["source_image"]) for item in self.data]
        self.qwen_ipa_embeddings_root = qwen_ipa_embeddings_root
        self.qwen_CA_embeddings_root = qwen_CA_embeddings_root
        self.qwen_text_embeddings_root = qwen_text_embeddings_root
        self.siglip_embeddings_root = siglip_embeddings_root
        self.clip_embeddings_root = clip_embeddings_root
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
      
        # 加载全0图特征（用于 dropout）
        self.zero_clip_embedding = None
        if self.clip_embeddings_root is not None:
            zero_image_path = os.path.join(self.clip_embeddings_root, "zero_image.safetensors")
            if os.path.exists(zero_image_path):
                self.zero_clip_embedding = load_file(zero_image_path)["embedding"]
        self.siglip_zero_embedding = None
        if self.siglip_embeddings_root is not None:
            siglip_zero_image_path = os.path.join(self.siglip_embeddings_root, "zero_image.safetensors")
            if os.path.exists(siglip_zero_image_path):
                self.siglip_zero_embedding = load_file(siglip_zero_image_path)["embedding"]
        # 加载''文本特征（用于 dropout）
        self.empty_qwen_text_embeddings = None
        if self.qwen_text_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_text_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_text_embeddings = load_file(empty_text_path)["embedding"]
        
        # 加载''文本特征（用于 dropout）
        self.empty_qwen_ipa_text_embeddings = None
        if self.qwen_ipa_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_ipa_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_ipa_text_embeddings = load_file(empty_text_path)["embedding"]

        # 加载''文本特征（用于 dropout）
        self.empty_qwen_ca_text_embeddings = None
        if self.qwen_CA_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_CA_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_ca_text_embeddings = load_file(empty_text_path)["embedding"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        '''
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])'''
        # 可视化保存相关参数
        self.vis_save_dir = "keshihua11"
        self.max_vis_samples = 2
        self.vis_counter = 0
        if self.vis_save_dir and self.max_vis_samples > 0:
            os.makedirs(self.vis_save_dir, exist_ok=True)
       


    def __getitem__(self, idx):
        item = self.data[idx]
        im_name = self.im_names[idx]
        parse_name = im_name.replace('.jpg', '.png')
        cloth_mask_name = im_name

        # 使用 BICUBIC 插值进行下采样，保持图像质量
        # 原始尺寸 1024x768 -> 目标尺寸 512x384，宽高比相同(0.75)，可直接resize
        s_vae_img = Image.open(os.path.join(self.image_root_path, item["source_image"])).resize(
            self.size, Image.BICUBIC)
        s_vae_img = self.transform(s_vae_img)
        t_vae_img = Image.open(os.path.join(self.image_root_path, item["target_image"])).resize(
            self.size, Image.BICUBIC)
        t_vae_img = self.transform(t_vae_img)

        #im_cloth_mask
        split = item["source_image"].split('/')[0]  # "train" or "test"
        cloth_mask_image = Image.open(os.path.join(
                self.image_root_path, split, 'cloth-mask', cloth_mask_name))
        processed_mask = self.mask_processor.preprocess(cloth_mask_image)
        inshop_mask = processed_mask.squeeze(0)
        




        #im_parse
        split = item["source_image"].split('/')[0]  # "train" or "test"
        im_parse = Image.open(os.path.join(
                self.image_root_path, split, 'image-parse-v3', parse_name))
        im_parse = im_parse.resize(
                self.size, Image.NEAREST)#保证分割的类别标签不被差值
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32)#所有的非背景像素1 背景区域变成全0
            
        parse_hair = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 4).astype(np.float32) + \
                         (parse_array == 13).astype(np.float32)

        parser_mask_fixed = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 18).astype(np.float32) + \
                                (parse_array == 19).astype(np.float32)

        parser_mask_changeable = (parse_array == 0).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + \
                   (parse_array == 15).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                          (parse_array == 6).astype(np.float32) + \
                          (parse_array == 7).astype(np.float32)
        parse_mask = (parse_array == 5).astype(np.float32) + \
                         (parse_array == 6).astype(np.float32) + \
                         (parse_array == 7).astype(np.float32)

        parser_mask_fixed = parser_mask_fixed + (parse_array == 9).astype(np.float32) + \
                                (parse_array == 12).astype(np.float32)

        parser_mask_changeable += np.logical_and(
                parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
     #   parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth).unsqueeze(0)  # 添加通道维度 (H, W) -> (1, H, W)
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(
        parse_shape, np.logical_not(parse_mask))
        im_mask_fine = s_vae_img * (1-parse_cloth)
        warp_cloth = s_vae_img * parse_cloth + (1 - parse_cloth)
      
        # 1. qwen_ipa embeddings
        if self.qwen_ipa_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_ipa_embeddings_root, split, f"{stem}.safetensors")
            qwen_ipa_embeddings = load_file(feature_path)["embedding"]    
         
        # 2. qwen_CA embeddings
        if self.qwen_CA_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_CA_embeddings_root, split, f"{stem}.safetensors")
            qwen_CA_embeddings = load_file(feature_path)["embedding"] 
       

        # 3. siglip embeddings
        if self.siglip_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.siglip_embeddings_root, split, f"{stem}.safetensors")
            siglip_s_img = load_file(feature_path)["embedding"]
        
        # 4. clip embeddings
        if self.clip_embeddings_root is not None:
           
            filename = os.path.basename(item["target_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["target_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.clip_embeddings_root, split, f"{stem}.safetensors")
            clip_t_img = load_file(feature_path)["embedding"]
        
        # 5. qwen_text embeddings
        if self.qwen_text_embeddings_root is not None:
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_text_embeddings_root, split, f"{stem}.safetensors")
            qwen_text_embeddings = load_file(feature_path)["embedding"]

        # 6. drop
        rand_num = random.random()
        if rand_num < 0.05:
            clip_t_img = self.zero_clip_embedding
            qwen_ipa_embeddings =self.empty_qwen_ipa_text_embeddings
            qwen_CA_embeddings = self.empty_qwen_ca_text_embeddings
        elif rand_num < 0.1:  
            siglip_s_img = self.siglip_zero_embedding
            qwen_text_embeddings = self.empty_qwen_text_embeddings
        elif rand_num < 0.15:  
            clip_t_img = self.zero_clip_embedding
            qwen_ipa_embeddings =self.empty_qwen_ipa_text_embeddings
            qwen_CA_embeddings = self.empty_qwen_ca_text_embeddings
            siglip_s_img = self.siglip_zero_embedding
            qwen_text_embeddings = self.empty_qwen_text_embeddings

        result = {
            "siglip_s_img": siglip_s_img,
            "im_mask_fine": im_mask_fine,
            "parse_cloth": parse_cloth,
            "s_vae_img": s_vae_img,
            "warp_cloth": warp_cloth,
            "qwen_ipa_embeddings": qwen_ipa_embeddings,
            "qwen_CA_embeddings": qwen_CA_embeddings,
            "clip_t_img": clip_t_img,
            "empty_qwen_text_embeddings": self.empty_qwen_text_embeddings,
            "qwen_text_embeddings": qwen_text_embeddings,
            "t_vae_img": t_vae_img,
            "inshop_mask": inshop_mask
        }

            
        
        return result

    def __len__(self):
        return len(self.data)


class ImageDatasetDressCode(Dataset):
    def __init__(
        self,

        image_root_path,
        json_file,
        size=(512,512),
        siglip_embeddings_root=None,
        qwen_ipa_embeddings_root=None,
        qwen_CA_embeddings_root=None,
        clip_embeddings_root=None,
        qwen_text_embeddings_root=None
        
    ):
        self.data = json.load(open(json_file))
        self.image_root_path = image_root_path

        self.size = size
        self.im_names = [os.path.basename(item["source_image"]) for item in self.data]
        self.qwen_ipa_embeddings_root = qwen_ipa_embeddings_root
        self.qwen_CA_embeddings_root = qwen_CA_embeddings_root
        self.qwen_text_embeddings_root = qwen_text_embeddings_root
        self.siglip_embeddings_root = siglip_embeddings_root
        self.clip_embeddings_root = clip_embeddings_root
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
      
        # 加载全0图特征（用于 dropout）
        self.zero_clip_embedding = None
        if self.clip_embeddings_root is not None:
            zero_image_path = os.path.join(self.clip_embeddings_root, "zero_image.safetensors")
            if os.path.exists(zero_image_path):
                self.zero_clip_embedding = load_file(zero_image_path)["embedding"]
        
        # 加载''文本特征（用于 dropout）
        self.empty_qwen_text_embeddings = None
        if self.qwen_text_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_text_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_text_embeddings = load_file(empty_text_path)["embedding"]
        
        # 加载''文本特征（用于 dropout）
        self.empty_qwen_ipa_text_embeddings = None
        if self.qwen_ipa_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_ipa_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_ipa_text_embeddings = load_file(empty_text_path)["embedding"]

        # 加载''文本特征（用于 dropout）
        self.empty_qwen_ca_text_embeddings = None
        if self.qwen_CA_embeddings_root is not None:
            empty_text_path = os.path.join(self.qwen_CA_embeddings_root, "empty_text_embedding.safetensors")
            if os.path.exists(empty_text_path):
                self.empty_qwen_ca_text_embeddings = load_file(empty_text_path)["embedding"]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        '''
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])'''
        # 可视化保存相关参数
        self.vis_save_dir = "keshihua11"
        self.max_vis_samples = 2
        self.vis_counter = 0
        if self.vis_save_dir and self.max_vis_samples > 0:
            os.makedirs(self.vis_save_dir, exist_ok=True)
       


    def __getitem__(self, idx):
        item = self.data[idx]
        im_name = self.im_names[idx]
        parse_name = im_name.replace('_0.jpg', '_4.png')
        cloth_mask_name = im_name.replace('_0.jpg', '_1.png')

        # 使用 BICUBIC 插值进行下采样，保持图像质量
        # 原始尺寸 1024x768 -> 目标尺寸 512x384，宽高比相同(0.75)，可直接resize
        s_vae_img = Image.open(os.path.join(self.image_root_path, item["source_image"])).resize(
            self.size, Image.BICUBIC)
        s_vae_img = self.transform(s_vae_img)
        t_vae_img = Image.open(os.path.join(self.image_root_path, item["target_image"])).resize(
            self.size, Image.BICUBIC)
        t_vae_img = self.transform(t_vae_img)

        #im_cloth_mask
        split = item["source_image"].split('/')[0]  # "upper" or "lower" or "dresses"
        cloth_mask_image = Image.open(os.path.join(
                self.image_root_path, split, 'masks', cloth_mask_name))
        processed_mask = self.mask_processor.preprocess(cloth_mask_image)
        inshop_mask = processed_mask.squeeze(0)
        




        #im_parse
        split = item["source_image"].split('/')[0]  # "upper" or "lower" or "dresses"
        category = split
        im_parse = Image.open(os.path.join(
                self.image_root_path, split, 'label_maps', parse_name))
        im_parse = im_parse.resize(
                self.size, Image.NEAREST)#保证分割的类别标签不被差值
        parse_array = np.array(im_parse)

        if category == 'dresses':
            label_cat = 7
            parse_cloth = (parse_array == 7).astype(np.float32)
        elif category == 'upper_body':
            label_cat = 4
            parse_cloth = (parse_array == 4).astype(np.float32)
        elif category == 'lower_body':
            label_cat = 6
            parse_cloth = (parse_array == 6).astype(np.float32)
        else: 
            raise ValueError(f"未知的类别: {category}")
    
        parse_cloth = torch.from_numpy(parse_cloth).unsqueeze(0)  # 添加通道维度 (H, W) -> (1, H, W)
        
        im_mask_fine = s_vae_img * (1-parse_cloth)
        warp_cloth = s_vae_img * parse_cloth + (1 - parse_cloth)
      
        # 1. qwen_ipa embeddings
        if self.qwen_ipa_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_ipa_embeddings_root, "train", f"{stem}.safetensors")
            qwen_ipa_embeddings = load_file(feature_path)["embedding"]    
         
        # 2. qwen_CA embeddings
        if self.qwen_CA_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_CA_embeddings_root, "train", f"{stem}.safetensors")
            qwen_CA_embeddings = load_file(feature_path)["embedding"] 
       

        # 3. siglip embeddings
        if self.siglip_embeddings_root is not None:
           
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.siglip_embeddings_root, "train", f"{stem}.safetensors")
            siglip_s_img = load_file(feature_path)["embedding"]
        
        # 4. clip embeddings
        if self.clip_embeddings_root is not None:
           
            filename = os.path.basename(item["target_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["target_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.clip_embeddings_root, "train", f"{stem}.safetensors")
            clip_t_img = load_file(feature_path)["embedding"]
        
        # 5. qwen_text embeddings
        if self.qwen_text_embeddings_root is not None:
            filename = os.path.basename(item["source_image"])  # "00000_00.jpg"
            stem = Path(filename).stem  # "00000_00"
            split = item["source_image"].split('/')[0]  # "train" or "test"
            feature_path = os.path.join(self.qwen_text_embeddings_root, "train", f"{stem}.safetensors")
            qwen_text_embeddings = load_file(feature_path)["embedding"]

        # 6. drop
        rand_num = random.random()
        if rand_num < 0.05:
            clip_t_img = self.zero_clip_embedding
        elif rand_num < 0.1:  
            qwen_ipa_embeddings =self.empty_qwen_ipa_text_embeddings
        elif rand_num < 0.15:  
            qwen_CA_embeddings = self.empty_qwen_ca_text_embeddings
        elif rand_num < 0.2:
            clip_t_img = self.zero_clip_embedding
            qwen_ipa_embeddings =self.empty_qwen_ipa_text_embeddings
            qwen_CA_embeddings = self.empty_qwen_ca_text_embeddings

        result = {
            "siglip_s_img": siglip_s_img,
            "im_mask_fine": im_mask_fine,
            "parse_cloth": parse_cloth,
            "s_vae_img": s_vae_img,
            "warp_cloth": warp_cloth,
            "qwen_ipa_embeddings": qwen_ipa_embeddings,
            "qwen_CA_embeddings": qwen_CA_embeddings,
            "clip_t_img": clip_t_img,
            "empty_qwen_text_embeddings": self.empty_qwen_text_embeddings,
            "qwen_text_embeddings": qwen_text_embeddings,
            "t_vae_img": t_vae_img,
            "inshop_mask": inshop_mask
        }
        

        return result

    def __len__(self):
        return len(self.data)


