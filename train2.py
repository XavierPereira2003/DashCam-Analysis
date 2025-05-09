import os
import json
import argparse
import torch
import torchvision
from PIL import Image
from transformers import DetrConfig, DetrImageProcessor, DetrForObjectDetection
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from safetensors.torch import save_file  # For saving model weights in safetensors format

# Custom dataset class for object detection using COCO annotations.
class IDDCocoDetection(torchvision.datasets.CocoDetection): 
    """
    A custom dataset class for loading and processing COCO-style datasets with 
    additional support for a processor and optional transformations.

    Args:
        img_folder (str): Path to the folder containing the images.
        annotations (str): Path to the COCO-style JSON annotations file.
        processor (callable): A processor function to encode images and annotations.
        transforms (callable, optional): Optional transformations to apply to the 
            images and targets. Defaults to None.

    """
    def __init__(self, img_folder, annotations, processor, transforms=None):
        super().__init__(img_folder, annFile=annotations, transforms=transforms)
        with open(annotations, 'r') as f:
            self.data = json.load(f)
        self.img_folder = img_folder
        self.processor = processor
        self.transforms = transforms
        self.images = {img["id"]: img for img in self.data["images"]}
        self.annotations = self._group_annotations_by_image()

    def _group_annotations_by_image(self):
        annotations_by_image = {}
        for ann in self.data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        return annotations_by_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = list(self.images.keys())[idx]
        img_info = self.images[image_id]
        img_path = img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        target = {"image_id": image_id, "annotations": self.annotations.get(image_id, [])}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # Remove batch dimension
        target = encoding["labels"][0]  # Remove batch dimension

        if self.transforms:
            img, target = self.transforms(img, target)
        return pixel_values, target

# Custom collate function to prepare batches.
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

# PyTorch Lightning module with DETR model initialized from scratch.
class Detr(pl.LightningModule):
    """
    A PyTorch Lightning module for training and validating the DETR (DEtection TRansformer) model 
    for object detection tasks, initialized from scratch.

    Args:
        lr (float): Learning rate for the optimizer.
        lr_backbone (float): Learning rate for the backbone parameters.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        num_labels (int): Number of object classes for detection.
    """
    
    def __init__(self, lr, lr_backbone, weight_decay, num_labels, id2label, label2id):
        super().__init__()
        
        # Create a configuration for the DETR model
        config = DetrConfig(
            num_labels=num_labels,
            id2label={str(k): v for k, v in id2label.items()},
            label2id=label2id,
        )
        
        # Initialize the model from scratch using the configuration
        self.model = DetrForObjectDetection(config)
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DETR model with PyTorch Lightning')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimiser')
    args = parser.parse_args()

    # Define dataset paths.
    train_img_folder = "idd20kII/leftImg8bit"
    val_img_folder = "idd20kII/leftImg8bit"
    train_ann_file = "train_output.json"
    val_ann_file = "val_output.json"

    # Initialize the DETR image processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Create the datasets.
    train_dataset = IDDCocoDetection(
        img_folder=train_img_folder,
        annotations=train_ann_file,
        processor=processor
    )
    val_dataset = IDDCocoDetection(
        img_folder=val_img_folder,
        annotations=val_ann_file,
        processor=processor
    )

    # Load custom category mappings from JSON files
    with open("id2label.json", 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    with open("label2id.json", 'r') as f:
        label2id = json.load(f)
    num_labels = len(label2id)

    # Create DataLoaders with the provided batch size.
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=7)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=7)

    # Initialize the model from scratch
    model = Detr(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # Initialize TensorBoard logger.
    logger = TensorBoardLogger("", name="logs")

    # Initialize ModelCheckpoint callback.
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="detr-{epoch:02d}-{validation_loss:.2f}",
        monitor="validation_loss",
        mode="min",
        save_top_k=2,
        verbose=True
    )

    # Initialize the Trainer with increased epochs since we're training from scratch
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gradient_clip_val=0.1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # Start training.
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    save_path = "models"
    os.makedirs(save_path, exist_ok=True)
    state_dict = model.model.state_dict()
    safetensors_path = os.path.join(save_path, "model.safetensors")
    save_file(state_dict, safetensors_path)

    # Save the id2label and label2id mappings
    with open(os.path.join(save_path, "id2label.json"), 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
        
    with open(os.path.join(save_path, "label2id.json"), 'w') as f:
        json.dump(label2id, f, indent=2)

    # Save the processor configuration
    processor.save_pretrained(save_path)

    print(f"Final model saved to '{safetensors_path}'.")
    print(f"Label mappings and processor configuration saved to '{save_path}'.")