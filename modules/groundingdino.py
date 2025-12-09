import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw

class GroundingDINO_Module:
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    device = "cpu"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    
    @classmethod
    def set_device(cls, device):
        """
        Set the default device for the class.
        
        Args:
            device (str): The device to use (e.g., 'cuda:0', 'cpu')
        """
        cls.device = device
        cls.model = cls.model.to(cls.device)
        return cls
    
    @classmethod
    def run(cls, image, text):
        if isinstance(image, str):
            image = Image.open(image)

        cls.set_device("cuda:0")
        
        inputs = cls.processor(images=image, text=text, return_tensors="pt").to(cls.device)
        
        with torch.no_grad():
            outputs = cls.model(**inputs)
            results = cls.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )

        # conver this to plain text that follows the following example.
        # The grounding expert found a leopard with score 0.53 and a panthera with score 0.32.

        answer_string = ""
        for result in results:
            answer_string += f"The grounding expert found a {result['label']} with score {result['score']}. "
        
        return answer_string
    
    @classmethod
    def visualize_detections(cls, image, results, class_colors=None):
        """
        Visualize detection results on the image.
        
        Args:
            image: PIL Image object
            results: Detection results from run method
            class_colors: Dictionary mapping class names to RGB color tuples
        
        Returns:
            PIL Image with drawn bounding boxes
        """
        if class_colors is None:
            class_colors = {
                "a cat": (255, 0, 0),  # Red for grapes
                "a remote control": (0, 128, 0)   # Green for olives
            }
            
        image_draw = image.copy()
        draw = ImageDraw.Draw(image_draw)
        
        detected_objects = {}
        
        # Extract results
        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        labels = results["labels"]
        
        for box, score, label in zip(boxes, scores, labels):
            # Get the label text
            
            # Count objects
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1
            
            # Draw bounding box
            x0, y0, x1, y1 = box
            color = class_colors.get(label.lower(), (255, 255, 0))  # Default to yellow
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # Add label and score
            text = f"{label}: {score:.2f}"
            draw.text((x0, y0-15), text, fill=color)
        
        # Print summary
        print("Detection Summary:")
        for obj, count in detected_objects.items():
            print(f"- Found {count} {obj}")
        
        image_draw.save('temp.jpg')

if __name__=="__main__":
    # image_path = "/nas-ssd2/dataset/coco2017/val2017/000000195918.jpg"
    image_path = "/nas-hdd/share_data/M3D-Cap/test_data/M3D_Cap/ct_quizze/005469/Axial_C__arterial_phase/20.png"
    # image_path = 'cats.jpg'
    GroundingDINO_Module.set_device("cuda:0")
    results = GroundingDINO_Module.run(image_path, "lesion. liver.")
    print(results)
    GroundingDINO_Module.visualize_detections(
        image=Image.open(image_path), 
        results=results
    )