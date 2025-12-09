from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from omegaconf import DictConfig

class Ovis2():
    def __init__(self, id="Ovis2-8B", model_name="AIDC-AI/Ovis2-8B", cfg: DictConfig = None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-8B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).to(int(cfg.device))
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    @torch.no_grad()
    def run_on_video(self, video, query, **kwargs):
        video = [Image.open(img_path).convert('RGB') for img_path in video]
        images_str = '\n'.join([f'Frame {i+1}: <image>' for i in range(len(video))]) + '\n'
        query = images_str + query
        max_partition = 4
        images = video

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output



    @torch.no_grad()
    def run(self, image, query, **kwargs):
        if isinstance(image, list):
            images_str = '\n'.join([f'Image {i+1}: <image>' for i in range(len(image))]) + '\n'
            query = images_str + query
            max_partition = 4
            images = image
        else:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            
            query = f'<image>\n{query}'
            max_partition = 9
            images = [image]

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True
        )
        if 'temperature' in kwargs:
            gen_kwargs['temperature'] = kwargs['temperature']
        if 'top_p' in kwargs:
            gen_kwargs['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            gen_kwargs['top_k'] = kwargs['top_k']
            

        with torch.inference_mode():
            
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output
    
    @torch.no_grad()
    def run_on_text(self, query):
        images = []
        max_partition = None
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pv.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device) if pv is not None else None for pv in [pixel_values]]

        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output