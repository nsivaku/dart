from modules.spacellava import SpaceLlava_Module as spatial
from modules.groundingdino import GroundingDINO_Module as grounder
from modules.yolo import YOLO_Module as detector
from modules.vlm import VLM_Module as vlm
# from modules.medical import Medical_Module as medical
from modules.qwenocr import QwenOCR_Module as ocr

expert_functions = {
    'ocr': lambda image, plain_text: ocr.run(image),
    'spatial': lambda image, prompt: spatial.run(image, prompt),
    'detector': lambda image:  detector.run(image),
    'grounder': lambda image, prompt: grounder.run(image, prompt),
    'captioning': lambda image, prompt: vlm.run(image, prompt, vlm_type='captioning'),
    'reasoner': lambda image, prompt: vlm.run(image, prompt, vlm_type='reasoning'),
    'attribute': lambda image, prompt: vlm.run(image, prompt, vlm_type='attribute'),
    # 'medical': lambda image, prompt: medical.run(image, prompt),
}