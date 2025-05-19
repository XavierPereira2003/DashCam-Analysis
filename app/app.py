import Models_Interface as Models_Interface

from typing import List
import argparse
import os
from glob import glob
import logging
from tqdm.auto import tqdm
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log", mode='w')],  # Truncate log file before each run
    force=True
)
logger = logging.getLogger(__name__)
logger.info("Starting DashCamAnalyzer")

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
    parser = argparse.ArgumentParser(description="Dash camera image analyser.\n Creates a report based on images in the folcer gievn.")
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the image folder to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output folder"
    )
    parser.add_argument(
        "--output_images",
        type=bool,
        default=True,
        help="Save images with detections"
    )

    args = parser.parse_args()

    images = extract_images(args.folder)
    detector= Models_Interface.DashCamAnalyzer()
    if not os.path.exists(args.folder):
        logger.error(f"Folder {str(args.folder)} does not exist.")
        sys.exit(1)
    print(f"Found {len(images)} images in {args.folder}")
    logger.info(f"Found {len(images)} images in {args.folder}")
    if len(images) == 0:
        print("No images found in the folder.")
        logger.info("No images found in the folder.")
        exit(0)
    else:
        os.makedirs(args.output_dir+"/images", exist_ok=True)
        try:
            for image in tqdm(images):
                msg = f"Processing image: {image}"
                logger.info(msg)
                try:
                    fig = detector.evaluate(image)
                    if args.output_images:
                        fig.savefig(os.path.join("output", "images", os.path.basename(image)))
                except Exception as e:
                    err_msg = f"Error processing image {image}: {e}"
                    logger.error(err_msg)
                    continue
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting gracefully.")
            sys.exit(0)
        except Exception as e:
            err_msg = f"Unexpected error: {e}"
            logger.error(err_msg)




