# Medieval Manuscript Layout Analysis

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-5.23.1-orange.svg)](https://gradio.app/)

> **A Gradio-based web application for manuscript layout analysis using three combined YOLO models: emanuskript, catmus, and zone.**

![Architecture](https://img.shields.io/badge/Models-3%20YOLO-green)
![Classes](https://img.shields.io/badge/Classes-25-purple)

---

## 🎯 Overview

This application provides a web-based interface for analyzing medieval manuscript images using three specialized ML models. It combines predictions from multiple models into a unified COCO-format output.

### Key Features

- **Multi-Model Inference**: Runs 3 YOLO models in parallel using multiprocessing
- **COCO Format Output**: Standard COCO JSON annotations for interoperability
- **Interactive Filtering**: Filter results by 25 predefined manuscript element classes
- **Batch Processing**: Process entire ZIP archives of images
- **Visual Overlays**: Color-coded annotations with semi-transparent fills
- **Statistics**: Per-image and aggregate statistics with graphs
- **Analytics Dashboard**: Built-in visitor tracking with GDPR compliance

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web Interface                      │
├─────────────┬─────────────────┬─────────────────────────────────┤
│ Single Image│  Batch (ZIP)    │         Analytics Tab           │
│    Tab      │     Tab         │                                 │
├─────────────┴─────────────────┴─────────────────────────────────┤
│              Parallel Processing Layer (spawn)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  emanuskript │  │    catmus    │  │     zone     │          │
│  │    model     │  │    model     │  │    model     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│              Combination & Filtering (ImageBatch)                │
├─────────────────────────────────────────────────────────────────┤
│                    COCO JSON Output                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/emanuskript/layout.git
cd layout

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Access the web interface at: `http://localhost:8000`

---

## 📦 Models

The application uses three YOLO models:

| Model | File | Purpose |
|-------|------|---------|
| **emanuskript** | `best_emanuskript_segmentation.pt` | Manuscript elements (scripts, initials, decorations) |
| **catmus** | `best_catmus.pt` | Lines and zones (DefaultLine, InterlinearLine) |
| **zone** | `best_zone_detection.pt` | Zone detection (MainZone, GraphicZone, etc.) |

---

## 🏷️ Supported Classes (25)

| Category | Classes |
|----------|---------|
| **Layout** | Border, Column, Table, Diagram |
| **Script** | Main script black, Main script coloured, Variant script black, Variant script coloured |
| **Initials** | Historiated, Inhabited, Zoo - Anthropomorphic, Embellished, Plain initial - coloured, Plain initial - Highlighted, Plain initial - Black |
| **Navigation** | Page Number, Quire Mark, Running header, Catchword |
| **Content** | Gloss, Illustrations |
| **Zones** | GraphicZone, MusicLine, MusicZone, Music |

---

## 💻 Usage

### Single Image Processing

1. Upload an image (JPEG/PNG/TIFF/BMP/WebP)
2. Adjust confidence threshold (default: 0.25)
3. Adjust IoU threshold (default: 0.3)
4. Select/deselect classes to visualize
5. Click **"Run 3 Models + Combine"**
6. Download results via buttons

### Batch Processing

1. Create a ZIP file with images (nested folders supported)
2. Upload the ZIP file
3. Configure thresholds and classes
4. Click **"Process ZIP"**
5. Browse results in gallery
6. Download merged annotations or complete ZIP

---

## ⚙️ Configuration

### Server Settings

```python
demo.launch(
    server_name="0.0.0.0",    # Bind to all interfaces
    server_port=8000,          # HTTP port
    max_threads=4,             # Concurrent request limit
)
```

### Multiprocessing

The application uses Python's `multiprocessing` with `spawn` method for safe parallel execution:

- 3 worker processes (one per model)
- Models are cached per-process after first load
- Pre-initialized pool avoids threading issues with Gradio

---

## 📁 Project Structure

```
layout/
├── app.py                              # Main Gradio application
├── test_combined_models.py             # Model inference logic
├── requirements.txt                    # Python dependencies
├── analytics.db                        # SQLite visitor database (auto-created)
├── best_emanuskript_segmentation.pt    # YOLO model weights
├── best_catmus.pt                      # YOLO model weights
├── best_zone_detection.pt              # YOLO model weights
├── utils/
│   └── image_batch_classes.py          # ImageBatch class & mappings
├── monitoring/                         # Grafana/Loki monitoring configs
└── APP_DOCUMENTATION.md                # Detailed documentation
```

---

## 📊 Output Format

Annotations are exported in **COCO format**:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 4,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, width, height],
      "area": 12345.0
    }
  ],
  "categories": [
    {"id": 4, "name": "Main script black"}
  ]
}
```

---

## 🔒 Privacy & Compliance

- **Anonymization**: IPs are hashed (SHA256, truncated)
- **Cookie Banner**: GDPR-compliant consent mechanism
- **Session Tracking**: Hash-based, no persistent identifiers

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📚 Documentation

For detailed documentation, see [APP_DOCUMENTATION.md](APP_DOCUMENTATION.md).
