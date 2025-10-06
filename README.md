# 🕳️ Pothole Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered pothole detection system using YOLOv8 segmentation model for real-time road damage assessment. This system can process single or multiple images, detect potholes with high accuracy, and generate comprehensive reports.

## 🎯 Features

- **🔍 High Accuracy Detection**: 74.2% mAP@0.5 with YOLOv8 segmentation model
- **📊 Batch Processing**: Process multiple images simultaneously
- **📝 Detailed Reports**: Generate comprehensive detection summaries
- **🏷️ Smart Renaming**: Automatically rename images with standardized format
- **🚀 Fast API**: RESTful API for web integration
- **📱 Real-time Processing**: 23 FPS inference speed

## 📈 Model Performance

Our YOLOv8n segmentation model achieves excellent performance metrics:

| Metric | Score | Grade |
|--------|-------|-------|
| **mAP@0.5** | 74.2% | B (Good) |
| **mAP@0.5:0.95** | 45.6% | Strict IoU |
| **Precision** | 68.1% | High accuracy |
| **Recall** | 73.4% | Detects most potholes |
| **F1-Score** | 70.7% | Balanced performance |
| **Inference Speed** | 43.6ms | 23 FPS |
| **Model Size** | 12.5 MB | Lightweight |

### 🎯 Performance Assessment
- **✅ Strengths**: High recall (73.4%) - detects most potholes, well-balanced precision and recall
- **🎯 Grade**: B (Good) - Ready for production deployment
- **⚡ Speed**: Moderate (23 FPS) - suitable for real-time applications

## 🖼️ Demo Results

### Before vs After Detection

**Original Image:**
```
Input: Raw road image with potholes
```

**After Detection:**
```
Output: Annotated image with bounding boxes around detected potholes
```

*Sample detection showing potholes highlighted with bounding boxes and confidence scores*

### 📊 Sample Detection Summary
```
POTHOLE DETECTION SUMMARY REPORT
========================================

Analysis Date: 2024-10-06 15:30:45
Confidence Threshold: 0.3

OVERALL STATISTICS:
--------------------
Total Images Processed: 15
Images with Potholes: 12
Images without Potholes: 3
Total Potholes Detected: 47
Average Potholes per Image: 3.13

DETAILED RESULTS:
--------------------
image_001.jpg (original: IMG_20240101.jpg): 2 potholes
image_002.jpg (original: photo_123.png): 0 potholes
image_003.jpg (original: road_damage.jpg): 5 potholes
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pothole-detection.git
cd pothole-detection
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the model**
   - Ensure `best.pt` is in the root directory
   - Model size: 12.5 MB

## 📖 Usage

### 🖼️ Image Detection (Local)

Process images from a local folder:

```bash
python test_image.py
```

**Interactive prompts:**
1. Enter folder path containing images
2. Set confidence threshold (0.1-0.9, default: 0.3)

**Output structure:**
```
pothole_results/
├── image_001.jpg          # Renamed + annotated
├── image_002.jpg          # Renamed + annotated
├── image_003.jpg          # Renamed + annotated
└── detection_summary.txt  # Detailed report
```

### 🌐 API Usage

Start the FastAPI server:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**

#### POST `/detect`
Upload single or multiple images for pothole detection.

**Parameters:**
- `file`: Image file (jpg/png)
- `conf`: Confidence threshold (0.1-0.9, default: 0.3)
- `return_image`: Return annotated image (boolean, default: false)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "file=@road_image.jpg" \
     -F "conf=0.3" \
     -F "return_image=true"
```

**Response (JSON):**
```json
{
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.85,
      "bbox": [[100, 150, 200, 250]]
    }
  ]
}
```

## 📁 Project Structure

```
pothole-detection/
├── 📄 README.md                    # This file
├── 🐍 app.py                       # FastAPI application
├── 🐍 test_image.py                # Local image processing
├── 🤖 best.pt                      # YOLOv8 model weights
├── 📋 requirements.txt             # Python dependencies
├── 📊 model_evaluation/            # Model analysis tools
│   ├── accuracy_analysis.py
│   ├── final_analysis.py
│   └── accuracy_results/
│       ├── accuracy_report.txt
│       └── accuracy_analysis.png
└── 📁 pothole_results/            # Output folder (generated)
    ├── image_001.jpg
    ├── image_002.jpg
    └── detection_summary.txt
```

## 🔧 Configuration

### Confidence Threshold Guidelines

| Threshold | Use Case | Precision | Recall |
|-----------|----------|-----------|---------|
| 0.2-0.4 | High Recall | Lower | Higher |
| 0.3-0.5 | **Balanced** ⭐ | Medium | Medium |
| 0.5-0.7 | High Precision | Higher | Lower |

### Model Specifications

- **Architecture**: YOLOv8n Segmentation
- **Input Size**: 640×640 pixels
- **Parameters**: 3.26M
- **Model Type**: Segmentation + Detection
- **Training**: Custom pothole dataset

## 📊 Model Evaluation

### Accuracy Metrics
Our model evaluation shows strong performance across key metrics:

- **Detection Accuracy**: 74.2% mAP@0.5
- **Localization Precision**: 45.6% mAP@0.5:0.95
- **False Positive Rate**: 31.9% (100% - 68.1% precision)
- **Miss Rate**: 26.6% (100% - 73.4% recall)

### Performance Benchmarks
- **Inference Time**: 43.6ms per image
- **Throughput**: 1,378 images per minute
- **Memory Usage**: ~500MB GPU memory
- **CPU Usage**: Compatible with CPU inference

## 🛠️ Development

### Adding New Features

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes and test**
4. **Submit pull request**

### Model Improvement

To improve model accuracy:

1. **Collect more training data** (2000+ images)
2. **Increase training epochs** (200-300)
3. **Upgrade model size** (YOLOv8s/YOLOv8m)
4. **Fine-tune hyperparameters**

Expected improvements: 85-90% mAP@0.5

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```bash
Error: [Errno 2] No such file or directory: 'best.pt'
```
**Solution**: Ensure `best.pt` is in the project root directory

**2. CUDA out of memory**
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU inference

**3. Import errors**
```bash
ModuleNotFoundError: No module named 'ultralytics'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

### Performance Optimization

- **GPU**: Use CUDA-compatible GPU for 5x speed improvement
- **Batch Processing**: Process multiple images simultaneously
- **Model Optimization**: Convert to TensorRT for 2-3x speed boost

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pothole-detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pothole-detection/discussions)
- **Email**: your.email@example.com

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv8 framework
- **FastAPI** for the web framework
- **OpenCV** for image processing
- **Contributors** who helped improve this project

---

⭐ **Star this repository if you found it helpful!**

*Built with ❤️ for safer roads*