import os
import requests
from PIL import Image
import torch
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText

def make_2x2_grid(imgs, margin=0, bg_color=(255,255,255), save=False):
    """
    Arrange four PIL images into a 2×2 grid.
    
    Parameters:
    - imgs: list or tuple of 4 PIL.Image instances [top-left, top-right, bottom-left, bottom-right]
    - margin: pixels of space between images (default 0)
    - bg_color: background color tuple for padding/margin (default white)
    
    Returns:
    - A new PIL.Image of the combined grid.
    """
    if len(imgs) != 4:
        raise ValueError("Expected exactly 4 images")
    
    # assume all images same size
    w, h = imgs[0].size

    # total grid size
    grid_w = 2 * w + margin
    grid_h = 2 * h + margin

    # create blank canvas
    grid_img = Image.new('RGB', (grid_w, grid_h), color=bg_color)

    # paste each image
    positions = [
        (0,     0),      # top-left
        (w+margin, 0),   # top-right
        (0,     h+margin),# bottom-left
        (w+margin, h+margin) # bottom-right
    ]
    for img, pos in zip(imgs, positions):
        grid_img.paste(img, pos)
    
    if save:
        grid_img.save(f"medgrid.png")

    return grid_img

class MedGemma:
    """
    A class-based wrapper around the google/medgemma-4b-it image-text-to-text pipeline
    for multimodal medical inference using only class methods.
    """
    _pipe = None
    _initialized = False

    @classmethod
    def initialize(cls,
                   model_name: str = "google/medgemma-4b-it",
                   torch_dtype=torch.bfloat16,
                   device: str = None):
        """Load the image-text-to-text pipeline once."""
        if cls._initialized:
            return

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create pipeline
        # cls._pipe = pipeline(
        #     task="text-generation",
        #     model=model_name,
        #     torch_dtype=torch_dtype,
        #     device=0 if device == "cuda" else -1,
        # )

        cls.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        cls.processor = AutoProcessor.from_pretrained(model_name)
        cls._initialized = True

    @classmethod
    def _load_image(cls, image_input):
        """
        Load an image from a URL, file path, or accept a PIL.Image directly.

        Args:
            image_input (str | PIL.Image.Image): URL, local path, or PIL image.

        Returns:
            PIL.Image.Image: RGB image.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                resp = requests.get(image_input, headers={"User-Agent": "medgemma-module"}, stream=True)
                resp.raise_for_status()
                img = Image.open(resp.raw)
                return img.convert("RGB")
            else:
                return Image.open(image_input).convert("RGB")
        if isinstance(image_input, list):
            # If a list of image paths, load them as a list of PIL images
            image = [Image.open(img_path).convert("RGB") for img_path in image_input]
            return make_2x2_grid(image, save=True)  # Combine into a 2x2 grid
        raise ValueError("Unsupported image_input type. Provide a URL, file path, or PIL.Image.")

    @classmethod
    def run_inference(cls,
                      image_input,
                      prompt: str,
                      max_new_tokens: int = 400) -> str:
        """
        Run inference on an image and text prompt.

        Args:
            image_input (str | PIL.Image.Image): URL/path or image.
            prompt (str): The medical question or instruction.
            max_new_tokens (int): Maximum tokens to generate.

        Returns:
            str: Generated response.
        """
        cls.initialize()

        # Load and prepare image
        

        # Build messages structure
        if image_input is not None:
            image = cls._load_image(image_input)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a medical expert observing a CT scan."}]},
                {"role": "user",   "content": [
                    {"type": "text",  "text": prompt},
                    {"type": "image", "image": image},
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a medical expert."}]},
                {"role": "user",   "content": [{"type": "text", "text": prompt}]},
            ]

        with torch.no_grad():
            inputs = cls.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(cls.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = cls.model.generate(**inputs, max_new_tokens=400, do_sample=False)
                generation = generation[0][input_len:]
            decoded = cls.processor.decode(generation, skip_special_tokens=True)
        return decoded

class MedGemmaModule:
    """
    High-level interface exposing only class methods for MedGemma inference.
    """

    @classmethod
    def run(cls, image_input, prompt: str) -> str:
        """Run the MedGemma multimodal model and return its answer."""
        return MedGemma.run_inference(image_input, prompt)

# Example usage
if __name__ == "__main__":
    images = ["/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/006007/Axial_C__arterial_phase/20.png", 
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/006007/Axial_C__arterial_phase/50.png",
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/006007/Axial_C__arterial_phase/80.png", 
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/006007/Axial_C__arterial_phase/110.png"]
    question = "Which organ exhibits multiple lacerations more than 1 cm in depth, as well as a small perinephric hematoma? Options: Liver, Spleen, Right kidney, Left kidney"
    # answer = MedGemmaModule.run(images, question)
    # print("Model answer:", answer)

    # 005512/Axial_bone_window
    # images = ["/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005512/Axial_bone_window/0.png", 
    #           "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005512/Axial_bone_window/33.png",
    #           "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005512/Axial_bone_window/66.png", 
    #           "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005512/Axial_bone_window/99.png"]
    # question = "Which window setting is used in this image?"
    answer = MedGemmaModule.run(images, question)
    print("Model answer:", answer)
