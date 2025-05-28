import os
import json
import torch
import matplotlib.pyplot as plt

from PIL import Image
import logging

from transformers import DetrImageProcessor, DetrForObjectDetection, logging as transformers_logging
from safetensors.torch import load_file

import warnings
# Suppress all warnings and HF transformer logs
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    
)
logger = logging.getLogger(__name__)


class DashCamAnalyzer:
    def __init__(self, device=None):
        """
        Initialize the DashCamAnalyzer with the model directory and device.

        Args:
            device (_type_, optional): Wheather the model should run on the CPU or GPU. Defaults to None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and label maps
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        model_dir = os.path.join(os.path.dirname(__file__), "Models")
        logger.debug(os.path.join(model_dir, "id2label.json"))

        self.id2label = json.load(open(os.path.join(model_dir, "id2label.json")))
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        self.label2id = json.load(open(os.path.join(model_dir, "label2id.json")))
        self.num_labels = len(self.id2label)

        # Load DETR model architecture
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True 
        )

        # Load fine‐tuned weights
        state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=False)

        # Remove gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"Model Succesfully loaded on {self.device}")
    
    def evaluate(self, image_path, generate_fig=True):
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        # Preprocess the image and make predictions
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        width, height = image.size
        postprocessed_outputs = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=0.9
        )
        results = postprocessed_outputs[0]
        return {"scores": results["scores"],
                "labels": results["labels"],
                "boxes": results['boxes']}
    
    def visualize(self, image_path, results):
        COLORS = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
            'orange', 'purple', 'lime', 'teal', 'brown', 'pink'
        ]
        image = Image.open(image_path).convert("RGB")
        scores, labels, boxes = results['scores'], results['labels'], results['boxes']
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        ax = plt.gca()
        colors = COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax), c in zip(
            scores.tolist(), labels.tolist(), boxes.tolist(), colors
        ):
            ax.add_patch(plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                fill=False, color=c, linewidth=3
            ))
            text = f'{self.model.config.id2label[label]}: {score:.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        return plt.gcf()
    
    def get_id2label(self):
        return self.id2label
    
    def get_label2id(self):
        return self.label2id
    
    def get_label(self, id):
        return self.id2label[id]