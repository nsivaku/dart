import io
import base64
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# Converts an image input (PIL Image or file path) into a base64 data URI
def image_to_base64_data_uri(image_input):
    if isinstance(image_input, str):
        with open(image_input, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported input type. Input must be a file path or a PIL.Image.Image instance.")
    return f"data:image/png;base64,{base64_data}"

hub_dir = '/nas-hdd/nsivaku/hf_hub/hub/models--remyxai--SpaceLLaVA/snapshots/84d5c1c0a6fb964ecb857ec24ad6c2fbe21fdb1e'

class Llava:
    def __init__(self, mmproj=f"mmproj-model-f16.gguf", model_path=f"ggml-model-q4_0.gguf", gpu=False):
        chat_handler = Llava15ChatHandler(clip_model_path=f'{hub_dir}/{mmproj}', verbose=True)
        n_gpu_layers = 0
        if gpu:
            n_gpu_layers = -1
        self.llm = Llama(model_path=f'{hub_dir}/{model_path}', chat_handler=chat_handler, n_ctx=2048, logits_all=True, n_gpu_layers=n_gpu_layers)

    def run_inference(self, image, prompt):
        data_uri = image_to_base64_data_uri(image)
        res = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        return res["choices"][0]["message"]["content"]

class SpaceLlava_Module:

    llm_model = Llava(gpu=True)

    @classmethod
    def run(cls, image, prompt):
        result = cls.llm_model.run_inference(image, prompt)
        return result.strip()

if __name__ == "__main__":
    image = "/nas-ssd2/dataset/coco2017/val2017/000000528578.jpg"
    prompt = "direction of flag waving, left, right, up, or down"
    result = SpaceLlava_Module.run(image, prompt)
    print(result)