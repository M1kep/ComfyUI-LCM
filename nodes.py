import folder_paths
from .lcm.lcm_scheduler import LCMScheduler
from .lcm.lcm_pipeline import LatentConsistencyModelPipeline
from .lcm.lcm_i2i_pipeline import LatentConsistencyModelImg2ImgPipeline
from os import path
import time
import torch
import random
import numpy as np
from comfy.model_management import get_torch_device

MAX_SEED = np.iinfo(np.int32).max


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


class LCM_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "positive_prompt": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, cfg, positive_prompt, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                scheduler=self.scheduler,
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            prompt=positive_prompt,
            width=width,
            height=height,
            guidance_scale=cfg,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="np",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        images_tensor = torch.from_numpy(result)

        return (images_tensor,)


class LCM_SamplerComfy:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json")
        )
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                    },
                ),
                "size": ("INT", {"default": 512, "min": 512, "max": 768}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                # "latent": ("LATENT",),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "conditioning": ("CONDITIONING",),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "torch_compile_mode": (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], {"default": "default"}),
                "diffusers_model":  (["HF(SimianLuo/LCM_Dreamshaper_v7)", *folder_paths.get_filename_list("diffusers")], ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, cfg, size, num_images, use_fp16, conditioning, torch_compile, torch_compile_mode, diffusers_model):
        if self.pipe is None:
            if not diffusers_model.startswith("HF"):
                diffusers_model_path = folder_paths.get_full_path("diffusers", diffusers_model)
            else:
                diffusers_model_path = diffusers_model[3:-1] # remove HF() and trailing parenthesis

            self.pipe = LatentConsistencyModelPipeline.from_pretrained(
                safety_checker=None,
                pretrained_model_name_or_path=diffusers_model_path,
                scheduler=self.scheduler,
                # custom_revision="main",
                # revision="fb9c5d167af11fd84454ae6493878b10bb63b067"
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(), torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(), torch_dtype=torch.float32)

            if torch_compile:
                self.pipe.unet = torch.compile(
                    self.pipe.unet, mode=torch_compile_mode, fullgraph=True
                )

        torch.manual_seed(seed)
        start_time = time.time()

        result = self.pipe(
            # prompt=positive_prompt,
            prompt_embeds=conditioning[0][0],
            width=size,
            height=size,
            guidance_scale=cfg,
            num_inference_steps=steps,
            # latents=latent["samples"].to(torch.float16),
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="latent",
        ).images

        print("LCM inference time: ", time.time() - start_time, "seconds")
        # images_tensor = torch.from_numpy(result)

        return ({"samples": result / 0.18215},)

class LCM_img2img_Sampler:
    def __init__(self):
        self.scheduler = LCMScheduler.from_pretrained(
            path.join(path.dirname(__file__), "scheduler_config.json"))
        self.pipe = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "images": ("IMAGE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "prompt_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                    "height": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "width": ("INT", {"default": 512, "min": 512, "max": 768}),
                    "num_images": ("INT", {"default": 1, "min": 1, "max": 64}),
                    "use_fp16": ("BOOLEAN", {"default": True}),
                    "positive_prompt": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, seed, steps, prompt_strength, cfg, images, positive_prompt, height, width, num_images, use_fp16):
        if self.pipe is None:
            self.pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(
                pretrained_model_name_or_path="SimianLuo/LCM_Dreamshaper_v7",
                safety_checker=None,
            )

            if use_fp16:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float16)
            else:
                self.pipe.to(torch_device=get_torch_device(),
                             torch_dtype=torch.float32)

        torch.manual_seed(seed)
        start_time = time.time()

        images = np.transpose(images, (0, 3, 1, 2))
        results = []
        for i in range(images.shape[0]):
            image = images[i]
            result = self.pipe(
                image=image,
                prompt=positive_prompt,
                strength=prompt_strength,
                width=width,
                height=height,
                guidance_scale=cfg,
                num_inference_steps=steps,
                num_images_per_prompt=num_images,
                lcm_origin_steps=50,
                output_type="np",
                ).images
            tensor_results = [torch.from_numpy(np_result) for np_result in result]
            results.extend(tensor_results)

        results = torch.stack(results)
        
        print("LCM img2img inference time: ", time.time() - start_time, "seconds")

        return (results,)


class DiffusersSelector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "diffusers_model": (folder_paths.get_filename_list("diffusers"),),
            },
        }

    RETURN_TYPES = (folder_paths.get_filename_list("diffusers"),)
    FUNCTION = "doit"
    CATEGORY = "sampling"

    def doit(self, diffusers_model):
        return (diffusers_model,)



NODE_CLASS_MAPPINGS = {
    "LCM_Sampler": LCM_Sampler,
    "LCM_SamplerComfy": LCM_SamplerComfy,
    "LCM_img2img_Sampler": LCM_img2img_Sampler,
    "Diffusers_Selector": DiffusersSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LCM_Sampler": "LCM Sampler",
    "LCM_SamplerComfy": "LCM Sampler(Comfy)",
    "Diffusers_Selector": "Diffusers Selector",
}
