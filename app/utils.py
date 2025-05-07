import os
import sys
import torch
from safetensors.torch import load_file
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
from PIL import ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.safetensors")

class DetrObjectDetector:
    def __init__(self):
        # Initialize the model architecture from the pretrained checkpoint.
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        # If the fine-tuned safetensors file exists, load its state dict.
        if os.path.exists(MODEL_PATH):
            state_dict = load_file(MODEL_PATH)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        # Initialize the image processor.
        self.image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def detect_objects(self, image):
        """
        Perform object detection on the provided image.
        Args:
            image: Input image (e.g., PIL Image or numpy array).
        Returns:
            The model outputs containing detection results.
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

if __name__ == '__main__':
    # If an image path is provided as an argument, load the image; otherwise, create an image with the text "this is the image".
    try:
        if len(sys.argv) > 1:
            img_path = sys.argv[1]
            image = Image.open(img_path)
        else:
            image = Image.new("RGB", (640, 480), color="white")
            draw = ImageDraw.Draw(image)
            draw.text((20, 220), "this is the image", fill="black")
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    try:
        detector = DetrObjectDetector()
        outputs = detector.detect_objects(image)
        print(outputs)
    except Exception as e:
        print(f"Error during object detection: {e}")
