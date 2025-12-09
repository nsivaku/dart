from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import torch

class Qwen():
    def __init__(self, id="Qwen2.5-7B-Instruct", model_name="Qwen/Qwen2.5-7B-Instruct", cfg: DictConfig = None):
        super().__init__()
        self.cfg = cfg

        model_name = "Qwen/Qwen2.5-14B-Instruct"
        print(cfg)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=int(cfg.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    @torch.no_grad() 
    def run(self, query):
        messages = [
            {"role": "system", "content": "You are a tool planning agent. Follow the user prompt and output only a JSON as directed.."},
            {"role": "user", "content": query}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.001
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response