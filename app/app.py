from typing import List
import models_int
import argparse
import os
from glob import glob

def extract_images(folder: str)->List[str]:
    """
    Find all images in a folder and return their paths.

    Args:
        path (_type_): _description_
    """
    patterns = ("**/*.jpg", "**/*.jpeg", "**/*.png")
    images = []
    for pattern in patterns:
        images.extend(glob(os.path.join(folder, pattern)))
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DashCamAnalyzer")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the image folder to analyze",
    )
    args = parser.parse_args()

    images = extract_images(args.folder)
    print(f"Found {len(images)} images in {args.folder}")


