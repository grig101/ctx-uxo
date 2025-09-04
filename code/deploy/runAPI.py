"""
UXO Detection API using FastAPI and ONNX Runtime.

This module provides a REST API for detecting unexploded ordnance in images
using a pre-trained Faster R-CNN model converted to ONNX format.
"""

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from io import BytesIO
from PIL import Image
import torch
import uvicorn
import argparse
import asyncio
import sys
import os
import time
from typing import List, Dict, Any, Optional
import logging

from tools.export_to_onnx import export_onnx, create_tensorrt_session
from model import create_model
from utils.device import get_device_info
from dataset import get_dataset_locations
from src import create_dataloader
from src.data.transforms import get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self, model_path: str, input_size: int, device: str = "auto", backbone: str = "resnet18", use_tensorrt: bool = True, tensorrt_precision: str = "FP16"):
        self.model_path = model_path
        self.input_size = input_size
        self.device = device
        self.backbone = backbone
        self.use_tensorrt = use_tensorrt
        self.tensorrt_precision = tensorrt_precision
        self.model = None
        self.onnx_session = None
        self.is_loaded = False
        
    async def load_model(self):
        """Load the model"""
        try:
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = create_model(num_classes=1, backbone=self.backbone)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device).eval()

            dummy_input = torch.randn((1, 3, self.input_size, self.input_size)).to(self.device)
            onnx_path = export_onnx(self.model, dummy_input, "./converted.onnx", optimize_for_tensorrt=True)
       
            # Create TensorRT session with specified precision
            if self.device == "cuda" and self.use_tensorrt:
                logger.info(f"Creating TensorRT session with {self.tensorrt_precision} precision")
                self.onnx_session = create_tensorrt_session(onnx_path, logger=logger, precision=self.tensorrt_precision)
            else:
                if self.device == "cuda":
                    logger.info("Using CUDA provider (TensorRT disabled)")
                    providers = [
                        ('CUDAExecutionProvider', {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                            'cudnn_conv_use_max_workspace': '1',
                            'do_copy_in_default_stream': '1',
                        }),
                        'CPUExecutionProvider'
                    ]
                else:
                    logger.info("Using CPU provider (TensorRT not available)")
                    providers = ["CPUExecutionProvider"]
                self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        try:
            img_resized = cv2.resize(image, (self.input_size, self.input_size))
            
            if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            img_normalized = img_resized.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_transposed, axis=0)
            
            return img_batch
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")
    
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform inference with proper error handling."""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            input_data = self.preprocess_image(image)
            
            input_name = self.onnx_session.get_inputs()[0].name
            
            start_time = time.time()
            outputs = self.onnx_session.run(None, {input_name: input_data})
            inference_time = time.time() - start_time
            
            boxes, labels, scores = outputs
            
            confidence_threshold = 0.5
            valid_indices = scores > confidence_threshold
            
            results = {
                "boxes": boxes[valid_indices].tolist(),
                "labels": labels[valid_indices].tolist(),
                "scores": scores[valid_indices].tolist(),
                "inference_time": inference_time,
                "num_detections": len(boxes[valid_indices])
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    global model_manager
    model_manager = None
    logger.info("API started successfully")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="UXO Detection API",
    description="API for detecting unexploded ordnance in images",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded if model_manager else False,
        "device": model_manager.device if model_manager else "unknown"
    }


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_path: str = "model.pth",
    input_size: int = 800,
    confidence_threshold: float = 0.5,
    backbone: str = "resnet18",
    use_tensorrt: bool = True,
    tensorrt_precision: str = "FP16"
):
    """
    Predict UXO in uploaded image.
    
    Args:
        file: Image file to analyze
        model_path: Path to model checkpoint
        input_size: Input size for the model
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        JSON with detection results
    """
    global model_manager
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model_manager is None:
        model_manager = ModelManager(model_path, input_size, backbone=backbone, use_tensorrt=use_tensorrt, tensorrt_precision=tensorrt_precision)
        await model_manager.load_model()
    
    try:
        image_bytes = await file.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)
        
        results = model_manager.infer(image_np)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch/")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_path: str = "model.pth",
    input_size: int = 800,
    backbone: str = "resnet18",
    use_tensorrt: bool = True,
    tensorrt_precision: str = "FP16"
):
    """
    Predict UXO in multiple images.
    
    Args:
        files: List of image files to analyze
        model_path: Path to model checkpoint
        input_size: Input size for the model
    
    Returns:
        JSON with detection results for all images
    """
    global model_manager
    
    if model_manager is None:
        model_manager = ModelManager(model_path, input_size, backbone=backbone, use_tensorrt=use_tensorrt, tensorrt_precision=tensorrt_precision)
        await model_manager.load_model()
    
    results = []
    
    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "file": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            inference_results = model_manager.infer(image_np)
            inference_results["file"] = file.filename
            results.append(inference_results)
            
        except Exception as e:
            results.append({
                "file": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


async def run_app(model_path: str, input_size: int, host: str = "0.0.0.0", port: int = 8000, backbone: str = "resnet18", use_tensorrt: bool = True, tensorrt_precision: str = "FP16"):
    """Run the FastAPI application."""
    global model_manager
    
    model_manager = ModelManager(model_path, input_size, backbone=backbone, use_tensorrt=use_tensorrt, tensorrt_precision=tensorrt_precision)
    
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UXO Detection API")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model checkpoint")
    parser.add_argument("--input_size", type=int, default=800, help="Input size for the model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture (resnet18, resnet50, mobilenet)")
    parser.add_argument("--use_tensorrt", action='store_true', default=True, help="Enable TensorRT optimization")
    parser.add_argument("--tensorrt_precision", type=str, default="FP16", choices=['FP16', 'FP32', 'INT8'], help="TensorRT precision mode")
    
    args = parser.parse_args()
    
    asyncio.run(run_app(args.model_path, args.input_size, args.host, args.port, args.backbone, args.use_tensorrt, args.tensorrt_precision))
