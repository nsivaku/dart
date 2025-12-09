import os
from huggingface_hub import login
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

from PIL import Image

def make_2x2_grid(imgs, margin=0, bg_color=(255,255,255)):
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

    return grid_img


class MedicalLlava:
    """
    A class-based wrapper around the ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1 model
    for multimodal medical inference using only class methods.
    """
    _model = None
    _tokenizer = None
    _initialized = False

    @classmethod
    def _login_hf(cls):
        """Authenticate to Hugging Face using HF_TOKEN environment variable."""
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
        login(hf_token)

    @classmethod
    def initialize(cls,
                   model_name: str = "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
                   device_map: int = 0,
                   torch_dtype=torch.float16):
        """Load the model and tokenizer once."""
        if cls._initialized:
            return
        cls._login_hf()
        cls._model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        cls._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        cls._initialized = True

    @classmethod
    def run_inference(cls,
                      image_input,
                      prompt: str,
                      temperature: float = 0.95,
                      streaming: bool = False) -> str:
        """
        Run inference on an image and text prompt.

        Args:
            image_input (str | PIL.Image.Image): Path to image file or PIL image.
            prompt (str): The medical question or instruction.
            temperature (float): Sampling temperature.
            streaming (bool): Whether to stream results.

        Returns:
            str or generator: Full response string or streaming generator.
        """
        cls.initialize()

        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, list):
            # If a list of image paths, load them as a list of PIL images
            image = [Image.open(img_path).convert("RGB") for img_path in image_input]
            image = make_2x2_grid(image)  # Combine into a 2x2 grid
        else:
            raise ValueError("Unsupported image_input type. Provide a file path or PIL Image.")

        # Prepare chat messages
        msgs = [
            {"role": "user", "content": prompt},
        ]

        # Call model
        response = cls._model.chat(
            image=image,
            msgs=msgs,
            tokenizer=cls._tokenizer,
            sampling=True,
            temperature=temperature,
            stream=streaming,
        )

        if streaming:
            return response

        # Collect streamed chunks
        answer = ""
        for chunk in response:
            answer += chunk
        return answer.strip()

class MedicalModule:
    """
    High-level interface exposing only class methods for easy inference.
    """

    @classmethod
    def run(cls, image_input, prompt: str) -> str:
        """Run the medical multimodal model and return a trimmed answer."""
        return MedicalLlava.run_inference(image_input, prompt)

# Example usage
if __name__ == "__main__":
    images = ["/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005375/Axial_C__delayed/0.png", 
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005375/Axial_C__delayed/26.png",
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005375/Axial_C__delayed/52.png", 
              "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005375/Axial_C__delayed/79.png"]
    question = "In which plane is the splenic abnormality most clearly visualized?"
    answer = MedicalModule.run(images, question)
    print("Model answer:", answer)
