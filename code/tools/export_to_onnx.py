"""
ONNX export utilities for UXO detection model.

This module provides functionality to export PyTorch models to ONNX format
for optimized inference with TensorRT FP16 support.
"""

import torch
import onnx
import onnxruntime as ort
import os
from typing import Optional


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, 
                onnx_path: str, logger=None, optimize_for_tensorrt: bool = True) -> str:
    """
    Export PyTorch model to ONNX format optimized for TensorRT FP16.
    
    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor
        onnx_path: Path to save ONNX model
        logger: Logger instance for output
        optimize_for_tensorrt: Whether to optimize for TensorRT conversion
        
    Returns:
        str: Path to exported ONNX model
    """
    try:
        model.eval()
        
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        
        opset_version = 16 if optimize_for_tensorrt else 11
        
        if isinstance(dummy_input, torch.Tensor):
            export_input = dummy_input
            input_names = ['input']
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'boxes': {0: 'num_detections'},
                'labels': {0: 'num_detections'},
                'scores': {0: 'num_detections'}
            }
        else:
            export_input = dummy_input
            input_names = ['input']
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'boxes': {0: 'num_detections'},
                'labels': {0: 'num_detections'},
                'scores': {0: 'num_detections'}
            }
        
        output_names = ['boxes', 'labels', 'scores']
        
        torch.onnx.export(
            model,
            export_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        if logger:
            logger.info(f"ONNX model exported to {onnx_path} with opset {opset_version}")
        else:
            print(f"ONNX model exported to {onnx_path} with opset {opset_version}")
            
        # Validate ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        if logger:
            logger.info(f"ONNX {onnx_path} model is valid.")
        else:
            print(f"ONNX {onnx_path} model is valid.")
            
        return onnx_path
        
    except Exception as e:
        if logger:
            logger.error(f"ONNX export failed: {e}")
        else:
            print(f"ONNX export failed: {e}")
        raise


def create_tensorrt_session(onnx_path: str, logger=None, precision: str = "FP16") -> ort.InferenceSession:
    """
    Create ONNX Runtime session optimized for TensorRT with FP16 precision.
    
    Args:
        onnx_path: Path to ONNX model
        logger: Logger instance for output
        precision: Precision mode ("FP16", "FP32", "INT8")
        
    Returns:
        ONNX Runtime InferenceSession with TensorRT provider
    """
    try:
        trt_provider_options = {
            'device_id': 0,
            'trt_max_workspace_size': 2147483648,  # 2GB
            'trt_fp16_enable': precision == "FP16",
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': './trt_cache/timing_cache',
            'trt_force_sequential_engine_build': False,
            'trt_max_partition_iterations': 10,
            'trt_min_subgraph_size': 1,
            'trt_dump_subgraphs': False,
            'trt_engine_decryption_enable': False,
            'trt_engine_decryption_lib_path': '',
            'trt_extra_plugin_lib_paths': '',
            'trt_profile_min_shapes': 'input:1x3x800x800',
            'trt_profile_max_shapes': 'input:4x3x800x800',
            'trt_profile_opt_shapes': 'input:1x3x800x800',
        }

        providers = [
            ('TensorrtExecutionProvider', trt_provider_options),
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                'cudnn_conv_use_max_workspace': '1',
                'do_copy_in_default_stream': '1',
            }),
            'CPUExecutionProvider'
        ]
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        if logger:
            logger.info(f"TensorRT session created with {precision} precision")
            logger.info(f"Available providers: {session.get_providers()}")
        else:
            print(f"TensorRT session created with {precision} precision")
            print(f"Available providers: {session.get_providers()}")
            
        return session
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create TensorRT session: {e}")
            logger.info("Falling back to CUDA provider")
        else:
            print(f"Failed to create TensorRT session: {e}")
            print("Falling back to CUDA provider")
        

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
        
        return ort.InferenceSession(onnx_path, providers=providers)