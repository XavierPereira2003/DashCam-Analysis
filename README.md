# DashCam Analysis

An advanced AI-powered dashcam image analysis system that performs object detection and generates comprehensive reports with an interactive web dashboard.

##  Features

- **Object Detection**: Fine-tuned DETR (Detection Transformer) model for identifying 41 different object classes
- **Batch Processing**: Analyze entire folders of dashcam images efficiently
- **Visual Reports**: Generate prediction images with bounding boxes and confidence scores
- **Interactive Dashboard**: Web-based interface for exploring analysis results
- **Performance Analytics**: Detailed statistics on detection confidence and class performance
- **Real-time Server**: Built-in web server for immediate dashboard access

##  Detected Object Classes

The system can detect 41 different classes including:
- **Vehicles**: Car, truck, bus, motorcycle, bicycle, trailer, train
- **Infrastructure**: Traffic lights, traffic signs, poles, buildings, bridges
- **Road Elements**: Road, sidewalk, curb, parking areas, guard rails
- **People & Animals**: Persons, riders, animals
- **Environment**: Sky, vegetation, walls, fences, tunnels

##  Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DashCam-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the analysis on a folder of images:

```bash
cd app
python app.py --folder /path/to/your/images --output_dir results
```

#### Command Line Options

- `--folder`: Path to folder containing dashcam images (required)
- `--output_dir`: Output directory for results (default: "output")
- `--device`: Device to run on ("cuda" or "cpu", auto-detected if not specified)
- `--generate_fig`: Generate prediction images with bounding boxes (default: False)

#### Example with all options:

```bash
python app.py --folder ./test --output_dir ./analysis_results --device cuda --generate_fig True
```

## Output

The analysis generates:

1. **JSON Reports**:
   - `results.json`: Raw detection results with scores and labels
   - `report.json`: Processed analytics and statistics

2. **Visual Content**:
   - Prediction images with bounding boxes (if `--generate_fig` is enabled)
   - Statistical charts and graphs

3. **Interactive Dashboard**:
   - Real-time web interface accessible via browser
   - Performance metrics and visualizations
   - Image gallery with predictions

4. **Analysis Reports**:
   - Confidence threshold distributions
   - Class performance rankings
   - Detection statistics

## Web Dashboard

After running the analysis, the system automatically starts a web server and displays access URLs:

```
DASHBOARD READY!
============================================================
 Output Directory: ./results
Web Server Port: 8000

 Access URLs:
   Local:    http://localhost:8000/detection_dashboard.html
   Local:    http://127.0.0.1:8000/detection_dashboard.html
   Network:  http://192.168.1.100:8000/detection_dashboard.html

 Press Ctrl+C to stop the server
============================================================
```

The dashboard includes:
- **Analysis Report**: Statistics, confidence distributions, class performance
- **Prediction Images**: Visual results with bounding boxes (if generated)

## Project Structure

```
DashCam-Analysis/
├── app/
│   ├── app.py                 # Main application script
│   ├── Models_Interface.py    # DETR model interface and analyzer
│   ├── Processor.py          # Data processing and report generation
│   ├── Models/               # Pre-trained model weights and labels
│   │   ├── model.safetensors
│   │   ├── id2label.json
│   │   └── label2id.json
│   └── Template/            # Web dashboard templates
│       ├── detection_dashboard.html
│       ├── styles.css
│       └── script.js
├── Models/                   # Additional model checkpoints
├── requirements.txt         # Python dependencies
└── README.md               # This file
```
## Analysis Capabilities

The system provides comprehensive analytics including:

- **Detection Statistics**: Total detections, confidence distributions
- **Class Performance**: Per-class detection counts and average confidence
- **Threshold Analysis**: Performance across different confidence levels
- **Visual Summaries**: Charts and graphs for easy interpretation


- Integrate additional model architectures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

