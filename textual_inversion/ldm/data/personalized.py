import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


import torch
import random

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

imagenet_templates_smallest = [
    'a photo of a {}',
]

imagenet_templates_small = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
    'an illustration of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'an illustration of a clean {}',
    'an illustration of a dirty {}',
    'a dark photo of the {}',
    'an illustration of my {}',
    'an illustration of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'an illustration of the {}',
    'a good photo of the {}',
    'an illustration of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'an illustration of the clean {}',
    'a rendition of a {}',
    'an illustration of a nice {}',
    'a good photo of a {}',
    'an illustration of the nice {}',
    'an illustration of the small {}',
    'an illustration of the weird {}',
    'an illustration of the large {}',
    'an illustration of a cool {}',
    'an illustration of a small {}',
    'a depiction of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a depiction of a clean {}',
    'a depiction of a dirty {}',
    'a dark photo of the {}',
    'a depiction of my {}',
    'a depiction of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a depiction of the {}',
    'a good photo of the {}',
    'a depiction of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a depiction of the clean {}',
    'a rendition of a {}',
    'a depiction of a nice {}',
    'a good photo of a {}',
    'a depiction of the nice {}',
    'a depiction of the small {}',
    'a depiction of the weird {}',
    'a depiction of the large {}',
    'a depiction of a cool {}',
    'a depiction of a small {}',
]

imagenet_dual_templates_small = [
    'a photo of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped photo of the {} with {}',
    'the photo of a {} with {}',
    'a photo of a clean {} with {}',
    'a photo of a dirty {} with {}',
    'a dark photo of the {} with {}',
    'a photo of my {} with {}',
    'a photo of the cool {} with {}',
    'a close-up photo of a {} with {}',
    'a bright photo of the {} with {}',
    'a cropped photo of a {} with {}',
    'a photo of the {} with {}',
    'a good photo of the {} with {}',
    'a photo of one {} with {}',
    'a close-up photo of the {} with {}',
    'a rendition of the {} with {}',
    'a photo of the clean {} with {}',
    'a rendition of a {} with {}',
    'a photo of a nice {} with {}',
    'a good photo of a {} with {}',
    'a photo of the nice {} with {}',
    'a photo of the small {} with {}',
    'a photo of the weird {} with {}',
    'a photo of the large {} with {}',
    'a photo of a cool {} with {}',
    'a photo of a small {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        if self.coarse_class_text:
            placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)

        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    

# class MVPersonalizedBase(Dataset):
#     def __init__(self,
#                  data_root,
#                  size=None,
#                  repeats=100,
#                  interpolation="bicubic",
#                  flip_p=0.0,
#                  set="train",
#                  placeholder_token="*",
#                  per_image_tokens=False,
#                  center_crop=False,
#                  mixing_prob=0.25,
#                  coarse_class_text=None,
#                  ):

#         self.data_root = data_root

#         self.image_paths = sorted([os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)])

#         # self._length = len(self.image_paths)
#         self.num_images = len(self.image_paths)
#         self._length = self.num_images 

#         self.placeholder_token = placeholder_token

#         self.per_image_tokens = per_image_tokens
#         self.center_crop = center_crop
#         self.mixing_prob = mixing_prob

#         self.coarse_class_text = coarse_class_text

#         if per_image_tokens:
#             assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

#         if set == "train":
#             self._length = self.num_images * repeats

#         self.size = size
#         self.interpolation = {"linear": PIL.Image.LINEAR,
#                               "bilinear": PIL.Image.BILINEAR,
#                               "bicubic": PIL.Image.BICUBIC,
#                               "lanczos": PIL.Image.LANCZOS,
#                               }[interpolation]
#         self.flip = transforms.RandomHorizontalFlip(p=flip_p)

#     def __len__(self):
#         return self._length

#     def __getitem__(self, i):
#         example = {}
#         image = Image.open(self.image_paths[i % self.num_images])

#         if not image.mode == "RGB":
#             image = image.convert("RGB")

#         placeholder_string = self.placeholder_token
#         if self.coarse_class_text:
#             placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

#         if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
#             text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
#         else:
#             text = random.choice(imagenet_templates_small).format(placeholder_string)


#         example["caption"] = text


#         # default to score-sde preprocessing
#         # img = np.array(image).astype(np.uint8)
#         # image = Image.fromarray(img)
#         # image = np.array(image).astype(np.uint8)
#         # example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        
#         image = np.array(image).astype(np.uint8)
#         top_left = image[:256, :256, :]     
#         top_right = image[:256, 256:, :]   
#         bottom_left = image[256:, :256, :]
#         bottom_right = image[256:, 256:, :]


#         sub_images = [
#             top_left, top_right,
#             bottom_left, bottom_right
#         ]
#         sub_images = [(img / 127.5 - 1.0).astype(np.float32) for img in sub_images]

#         example["image"] = np.stack(sub_images, axis=0)  #  Input tensor shape: torch.Size([1, 4, 256, 256, 3]). Additional info: {}.
#         return example  
    


class MVGSPersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=1000,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):

        self.data_root = data_root

        self.num_images = 1
        self._length = 1

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        self.coarse_class_text = coarse_class_text

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size


        ### novel view (manual batch)
        self.device = torch.device("cuda")
        self.render_resolution = 256
        self.min_ver = -30
        self.max_ver = 30
        self.elevation = 0
        self.radius = 2.5
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        self.min_ver = max(min(self.min_ver, self.min_ver - self.elevation), -80 - self.elevation)
        self.max_ver = min(max(self.max_ver, self.max_ver - self.elevation), 80 - self.elevation)
        self.render = 'render'
        # self.bg_color = torch.tensor([1, 1, 1])
        self.bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

        self.cam = OrbitCamera(800, 800, r=2.5, fovy=49.1)
        self.renderer = Renderer(sh_degree=0)
        self.renderer.initialize(data_root)  

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        placeholder_string = self.placeholder_token
        # if self.coarse_class_text:
        #     placeholder_string = f"{self.coarse_class_text} {placeholder_string}"

        if self.per_image_tokens and np.random.uniform() < self.mixing_prob:
            text = random.choice(imagenet_dual_templates_small).format(placeholder_string, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(placeholder_string)
        # text = placeholder_string
        
        text = placeholder_string
        example["caption"] = text


        images = []
        poses = []

        # render random view
        ver = np.random.randint(self.min_ver, self.max_ver)
        hor = np.random.randint(-180, 180)
        radius = 0
        
        for view_i in range(0, 4):

            pose_i = orbit_camera(self.elevation + ver, hor + 90 * view_i, self.radius + radius)
            poses.append(pose_i)

            cur_cam_i = MiniCam(pose_i, self.render_resolution, self.render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)


            # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
            out_i = self.renderer.render(cur_cam_i, bg_color=self.bg_color)

            image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

            images.append((image * 2 - 1.0))

        images = torch.cat(images, dim=0)
        poses = np.stack(poses, axis=0)
        example['image'] = images
        example['poses'] = poses

        return example  