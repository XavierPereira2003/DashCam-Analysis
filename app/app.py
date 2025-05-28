import Models_Interface as Models_Interface
from Processor import Processor

from typing import List
import argparse
import os
from glob import glob
import logging
from tqdm.auto import tqdm
import sys
import warnings
import json
import shutil # Added import
import socket
import threading
import http.server
import socketserver
from urllib.parse import urljoin

# Suppress all warnings
warnings.filterwarnings("ignore")



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

def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"

def find_free_port(start_port=8000):
    """Find a free port starting from start_port."""
    port = start_port
    while port < start_port + 100:  # Try up to 100 ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    return None

def start_web_server(output_dir, port):
    """Start a web server in a separate thread."""
    class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress HTTP request logs
            pass
    
    def run_server():
        os.chdir(output_dir)
        with socketserver.TCPServer(("", port), QuietHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread

def display_dashboard_info(output_dir, port):
    """Display dashboard access information."""
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("DASHBOARD READY!")
    print("="*60)
    print(f" Output Directory: {output_dir}")
    print(f"Web Server Port: {port}")
    print("\n Access URLs:")
    print(f"   Local:    http://localhost:{port}/detection_dashboard.html")
    print(f"   Local:    http://127.0.0.1:{port}/detection_dashboard.html")
    print(f"   Network:  http://{local_ip}:{port}/detection_dashboard.html")

    print("\n Press Ctrl+C to stop the server")
    print("="*60)


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
        "--device",
        type=str,
        default=None,
        help="Device to run the model on. If not specified, it will use the default device (GPU if available, otherwise CPU)."
    )

    parser.add_argument(
        "--generate_fig",
        type=bool,
        default=False,
        help="Generate figure with detections"
    )
    args = parser.parse_args()

    images = extract_images(args.folder)
    detector= Models_Interface.DashCamAnalyzer(device=args.device)
    
    if not os.path.exists(args.folder):
        logger.error(f"Folder {str(args.folder)} does not exist.")
        sys.exit(1)
    print(f"Found {len(images)} images in {args.folder}")
    logger.info(f"Found {len(images)} images in {args.folder}")

    if len(images) == 0:
        print("No images found in the folder.")
        logger.info("No images found in the folder.")
        exit(0)
    results=[]
    generated_images = []  # Track generated prediction images
    try:
        os.makedirs(args.output_dir, exist_ok=True)

        

        for image in tqdm(images):
            msg = f"Processing image: {image}"
            logger.info(msg)
            try:
                result = detector.evaluate(image)
                results.append({"scores": result["scores"].tolist(), "labels": [detector.get_label(label) for label in result["labels"].tolist()]})
                if args.generate_fig:
                    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
                    fig = detector.visualize(image, result)
                    output_image_path = os.path.join(args.output_dir, "images", os.path.basename(image))
                    fig.savefig(output_image_path)
                    generated_images.append(os.path.basename(image))
                    logger.info(f"Generated prediction figure: {os.path.basename(image)}")
            except Exception as e:
                err_msg = f"Error processing image {image}: {e}"
                logger.error(err_msg)
                continue
        logger.info("Processing complete.")
        print("Processing complete.")
        print("Generating Report...")
        processor=Processor(data=results)
        report=processor.generate_report(os.path.join(args.output_dir,"report"))
        
        # Add generated images list to the report
        report['generated_images'] = generated_images
        report['has_prediction_images'] = len(generated_images) > 0
        
        print("Report generated successfully.")
        logger.info("Report generated successfully.")
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        with open(os.path.join(args.output_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=4)


        template_dir = os.path.join(os.path.dirname(__file__), "Template")
        if os.path.isdir(template_dir):
            for item_name in os.listdir(template_dir):
                source_item_path = os.path.join(template_dir, item_name)
                destination_item_path = os.path.join(args.output_dir, item_name)
                if os.path.isfile(source_item_path):
                    shutil.copy2(source_item_path, destination_item_path)
                    logger.info(f"Copied template file {item_name} to {args.output_dir}")
                
        else:
            logger.warning(f"Template directory {template_dir} not found. Skipping template copy.")
            
        # Start web server and display dashboard info
        port = find_free_port(8000)
        if port:
            start_web_server(args.output_dir, port)
            display_dashboard_info(args.output_dir, port)
            
            # Keep the server running
            try:
                while True:
                    input()  # Keep the main thread alive
            except KeyboardInterrupt:
                print("\nServer stopped by user.")
                sys.exit(0)
        else:
            print(" Could not find a free port to start the web server.")
            print(f"You can manually open: {os.path.join(args.output_dir, 'detection_dashboard.html')}")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)



