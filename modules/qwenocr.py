from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

class QwenOCR_Module:
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    device = "cpu"
    
    # Initialize model and processor as class attributes
    model = None
    processor = None
    
    @classmethod
    def _initialize_model(cls):
        """Initialize the model and processor if not already done."""
        if cls.model is None or cls.processor is None:
            cls.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cls.model_id,
                torch_dtype="auto",
                device_map="auto"
            )
            cls.processor = AutoProcessor.from_pretrained(cls.model_id)
    
    @classmethod
    def set_device(cls, device):
        """
        Set the default device for the class.
        
        Args:
            device (str): The device to use (e.g., 'cuda:0', 'cpu')
        """
        cls.device = device
        cls._initialize_model()
        cls.model = cls.model.to(cls.device)
        return cls
    
    @classmethod
    def run(cls, image, prompt="Extract the text from this image."):
        """
        Extract text from an image using Qwen OCR model.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for OCR task (default: "Extract the text from this image.")
        
        Returns:
            str: Extracted text from the image
        """
        # Initialize model if not already done
        cls._initialize_model()
        cls.set_device("cuda:3")
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Process the input
        text = cls.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = cls.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(cls.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = cls.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = cls.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        return output_text[0]

# if __name__ == "__main__":
#     # Example usage
#     image_path = "path/to/your/document_image.jpg"
#     QwenOCR_Module.set_device("cuda" if torch.cuda.is_available() else "cpu")
#     extracted_text = QwenOCR_Module.run(image_path)
#     print("Extracted Text:", extracted_text)
