# CTX-UXO: Contextual Visual Dataset for Unexploded Ordnance Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

#### National University of Science and Technology Politehnica of Bucharest

## Paper
**CTX-UXO: Contextual visual dataset for field-based localisation, classification and identification of unexploded ordnance**

## Dataset
The CTX-UXO dataset contains 3520 high-resolution RGB images with 15,449 annotated UXO instances across 12 subcategories.

### Download
The framework automatically downloads the dataset when not found locally. Dataset will be downloaded from Hugging Face or Zenodo and extracted automatically.

**Dataset Structure:**
```
./dataset/ctxuxo/
├── data.yaml
├── train/
├── valid/
└── test/
```

### Manual Download Links:
- [Hugging Face](https://huggingface.co/datasets/UXO-Politehnica-Bucharest/Contextual_Vision_for_Unexploded_Ordnances) 
- [DataPort](https://ieee-dataport.org/documents/ctx-uxo-comprehensive-dataset-detection-and-identification-unexploded-ordnances)
- [Zenodo](http://link.com/)

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA 12.0+ (for GPU training)
- 8GB+ VRAM (MobileNet), 15GB+ VRAM (ResNet models)
- TensorRT 8.0+ (for optimized inference)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/ctx-uxo.git
cd ctx-uxo
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt

# CPU only
pip install -r requirements_cpu.txt
```

## Usage

### Training

```bash
# Basic training with MobileNet
python main.py train

# Training with specific backbone
python main.py train --backbone mobilenet --batch_size 16 --learning_rate 4e-3

# Training with ResNet50
python main.py train --backbone resnet50 --batch_size 32 --learning_rate 8e-3

# Custom parameters
python main.py train \
    --num_epochs 100 \
    --batch_size 16 \
    --learning_rate 4e-3 \
    --input_size 800 \
    --copy_paste_probability 0.6 \
    --device cuda
```

### Testing

```bash
# Test with TensorRT FP16
python main.py test --use_tensorrt --tensorrt_precision FP16

# Test with custom model
python main.py test --model_path ./results/last_checkpoint.pth --use_tensorrt

# Test without TensorRT
python main.py test --use_tensorrt false
```

### Deployment

```bash
# Deploy API with TensorRT
python main.py deploy --model_path ./results/last_checkpoint.pth --use_tensorrt

# Using Docker
docker compose -f docker-compose.deploy.yaml up --build
```

## Configuration

### Model Configuration
```yaml
model:
  input_size: 800
  num_classes: 1
  backbone: "mobilenet"  # resnet18, resnet50, mobilenet
  pretrained: true
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 4e-3
  weight_decay: 1e-5
  momentum: 0.9
  scheduler: true
  warmup_epochs: 4
  early_stopping_epochs: 40
  seed: 41
```

### Testing Configuration
```yaml
testing:
  model_path: "./results/last_checkpoint.pth"
  input_size: 800
  batch_size: 1
  device: "auto"
  confidence_threshold: 0.05
  use_tensorrt: true
  tensorrt_precision: "FP16"
```

## Project Structure

```
code/
├── main.py               # Main CLI interface
├── config.yaml           # Configuration file
├── requirements.txt      # GPU dependencies
├── requirements_cpu.txt  # CPU dependencies
├── model/                # Model definitions
│   └── faster_rcnn.py   # Faster R-CNN implementation
├── tools/                # Training/testing tools
│   ├── train.py         # Training pipeline
│   ├── test.py          # Testing with TensorRT
│   ├── val.py           # Validation
│   └── export_to_onnx.py # ONNX export
├── utils/                # Utility functions
├── src/                  # Data loading
├── deploy/               # Deployment API
└── dataset/              # Dataset files
```


## API Usage

### Single Image Prediction
```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg" \
     -F "use_tensorrt=true" \
     -F "tensorrt_precision=FP16"
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Docker Deployment

```bash
docker compose -f docker-compose.deploy.yaml up --build
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir ./logs
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@misc{ctx-uxo-2025,
    title={CTX-UXO: Contextual visual dataset for field-based localisation, classification and identification of unexploded ordnance},
    author={M. Craioveanu et. al},
    year={2025},
    eprint={},
    archivePrefix={},
    primaryClass={}
}
```

## Troubleshooting

### Common Issues

**CUDA out of memory**
- Use MobileNet backbone
- Reduce batch size
- Enable TensorRT FP16

**TensorRT not working**
- Ensure TensorRT 8.0+ is installed
- Use fallback: `--use_tensorrt false`

**Model not loading**
- Check model path in config.yaml
- Verify checkpoint file exists

**API not responding**
- Check if port 8000 is available
- Verify model path is correct

**Slow inference**
- Enable TensorRT FP16
- Use MobileNet backbone
- Check GPU memory usage
