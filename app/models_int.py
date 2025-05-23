import os
import json
import torch
from PIL import Image
import json
from transformers import DetrImageProcessor, DetrForObjectDetection
from safetensors.torch import load_file
import matplotlib.pyplot as plt

class DashCamAnalyzer:
    def __init__(self, model_dir="Models", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load processor and label maps
        self.processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50"
        )
        print(os.path.join(model_dir, "id2label.json"))
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
    
    def evaluate(self, image_path):
        print(self.model.config.id2label)
        COLORS = [
            'red', 'green', 'blue', 'yellow', 'magenta', 'cyan',
            'orange', 'purple', 'lime', 'teal', 'brown', 'pink'
        ]
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


