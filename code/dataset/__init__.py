import os
import yaml
import json
from pathlib import Path
from .download.get_hf import download_hf_dataset, download_dataset_with_config
from .download.get_zenodo import download_zd_dataset

def build_paths(images_path_str, yaml_dir):
    if images_path_str is None:
        return None

    images_path = (yaml_dir / images_path_str).resolve()
    dataset_root = images_path.parent

    return {
        "images": str(images_path),
        "labels": str(dataset_root / "labels")
    }

def get_dataset_locations(yaml_path, config=None):
    """
    Get dataset locations from YAML file, with automatic download if not found.
    
    Args:
        yaml_path: Path to the data.yaml file
        config: Optional configuration dictionary for dataset settings
        
    Returns:
        dict: Dataset paths for train, val, and test splits
    """
    yaml_path = Path(yaml_path).resolve()

    if not yaml_path.is_file():
        print(f"YAML file not found at: {yaml_path}")
        print("Attempting to download dataset automatically...")
        
        try:
            dataset_dir = yaml_path.parent
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            yaml_path = dataset_dir / "data.yaml"
            
            if not yaml_path.exists():
                if config and 'dataset' in config:
                    download_hf_dataset(
                        dataset_name=config['dataset'].get('hf_name', "UXO-Politehnica-Bucharest/Contextual_Vision_for_Unexploded_Ordnances"),
                        dest_dir=str(dataset_dir),
                        auto_extract=config['dataset'].get('auto_extract', True),
                        cleanup_archives=config['dataset'].get('cleanup_archives', True),
                        reorganize_structure=config['dataset'].get('reorganize_structure', True)
                    )
                else:
                    download_hf_dataset(dest_dir=str(dataset_dir))
            
            if yaml_path.exists():
                with yaml_path.open('r') as f:
                    data = yaml.safe_load(f)
                
                yaml_dir = yaml_path.parent
                dataset_paths = {
                    "train": build_paths(data.get("train"), yaml_dir),
                    "val": build_paths(data.get("val"), yaml_dir),
                    "test": build_paths(data.get("test"), yaml_dir)
                }
                print("Dataset paths:\n")
                print(json.dumps(dataset_paths, indent=4))
                return dataset_paths
            else:
                raise FileNotFoundError(f"Dataset YAML file not found at {yaml_path}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise

    with yaml_path.open('r') as f:
        data = yaml.safe_load(f)

    yaml_dir = yaml_path.parent

    dataset_paths = {
        "train": build_paths(data.get("train"), yaml_dir),
        "val": build_paths(data.get("val"), yaml_dir),
        "test": build_paths(data.get("test"), yaml_dir)
    }
    print("Dataset paths:\n")
    print(json.dumps(dataset_paths, indent=4))
    return dataset_paths