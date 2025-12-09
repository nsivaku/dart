from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
from omegaconf import DictConfig

class MiniCPM_o():
    def __init__(self, id="MiniCPM-o-2_6", model_name="openbmb/MiniCPM-o-2_6", cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=True
        ).eval().to(int(cfg.device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # self.model.init_tts()

    @torch.no_grad()
    def run_on_video(self, video, query):
        video = [Image.open(img_path).convert('RGB') for img_path in video]
        msgs = [
            {'role': 'user', 'content': video + [query]}, 
        ]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return res
        
    @torch.no_grad() 
    def run(self, image, query):
        if isinstance(image, list):
            msgs = [{
                    "role": "user",
                    "content":['What fruit is in the image?\nAnswer the question with one word or short phrase then explain with step-by-step reasoning.\n\nYou must output in the following format:\nAnswer: [answer]\nReasoning: [reasoning]"']
                },
                {
                    "role": "assistant",
                    "content": ["Answer: apple\nReasoning: The image shows a red fruit with a stem and leaves, which is characteristic of an apple. Apples are commonly red, green, or yellow and have a round shape."]
                },
                {'role': 'user', 'content': [*image, query]}]
            res = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer
            )
            return res
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        msgs = [{'role': 'user', 'content': [image, query]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return res
    
    def run_on_text(self, query):
        msgs = [{'role': 'user', 'content': [query]}]
        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer
        )
        return res
