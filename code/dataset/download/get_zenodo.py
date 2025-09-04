import os
import sys
import zipfile
import tarfile
import shutil
import requests
from pathlib import Path
import logging

def reorganize_dataset_structure(dataset_dir):
    """
    Reorganize the dataset structure after extraction.
    
    The expected structure after extraction is:
    ./dataset/ctxuxo/
    ├── coco_labels/
    ├── images/
    ├── yolo_bbox/
    └── yolo_segmentation/
        ├── data.yaml
        ├── train/
        ├── valid/
        └── test/
    
    We want to reorganize it to:
    ./dataset/ctxuxo/
    ├── data.yaml (moved from yolo_segmentation/)
    ├── train/ (merged from images/train and yolo_segmentation/train)
    ├── valid/ (merged from images/valid and yolo_segmentation/valid)
    └── test/ (merged from images/test and yolo_segmentation/test)
    
    Args:
        dataset_dir: Path to the dataset directory
        
    Returns:
        bool: True if reorganization successful, False otherwise
    """
    try:
        dataset_path = Path(dataset_dir)
        
        print("Reorganizing dataset structure...")
        
        yolo_seg_dir = dataset_path / "yolo_segmentation"
        if not yolo_seg_dir.exists():
            print("Warning: yolo_segmentation directory not found. Skipping reorganization.")
            return True
        
        yaml_source = yolo_seg_dir / "data.yaml"
        yaml_dest = dataset_path / "data.yaml"
        
        if yaml_source.exists():
            print(f"Moving data.yaml from {yaml_source} to {yaml_dest}")
            shutil.move(str(yaml_source), str(yaml_dest))
        else:
            print("Warning: data.yaml not found in yolo_segmentation directory")
        
        for split in ['train', 'valid', 'test']:
            images_split_dir = dataset_path / "images" / split
            yolo_split_dir = yolo_seg_dir / split
            
            dest_split_dir = dataset_path / split
            
            print(f"Processing {split} split...")
            
            dest_split_dir.mkdir(exist_ok=True)
            
            if images_split_dir.exists():
                print(f"  Copying from {images_split_dir} to {dest_split_dir}")
                for item in images_split_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(str(item), str(dest_split_dir / item.name))
                    elif item.is_dir():
                        shutil.copytree(str(item), str(dest_split_dir / item.name), dirs_exist_ok=True)
            
            if yolo_split_dir.exists():
                print(f"  Copying from {yolo_split_dir} to {dest_split_dir}")
                for item in yolo_split_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(str(item), str(dest_split_dir / item.name))
                    elif item.is_dir():
                        shutil.copytree(str(item), str(dest_split_dir / item.name), dirs_exist_ok=True)
        
        print("Cleaning up old directories...")
        
        if yolo_seg_dir.exists():
            shutil.rmtree(str(yolo_seg_dir))
            print(f"  Removed {yolo_seg_dir}")
        
        images_dir = dataset_path / "images"
        if images_dir.exists():
            shutil.rmtree(str(images_dir))
            print(f"  Removed {images_dir}")
        
        for dir_name in ['coco_labels', 'yolo_bbox']:
            dir_path = dataset_path / dir_name
            if dir_path.exists():
                shutil.rmtree(str(dir_path))
                print(f"  Removed {dir_path}")
        
        print("Dataset reorganization completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error reorganizing dataset structure: {e}")
        return False

def extract_archive(archive_path, extract_to):
    """
    Extract archive file to destination directory.
    
    Args:
        archive_path: Path to the archive file
        extract_to: Destination directory for extraction
        
    Returns:
        bool: True if extraction successful, False otherwise
    """
    try:
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        
        print(f"Extracting archive: {archive_path}")
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
            
        print(f"Archive extracted successfully to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

def find_and_extract_archives(directory, auto_extract=True, cleanup_archives=True):
    """
    Find and extract all archive files in the given directory.
    
    Args:
        directory: Directory to search for archives
        auto_extract: Whether to automatically extract archives
        cleanup_archives: Whether to remove archive files after extraction
        
    Returns:
        bool: True if all extractions successful, False otherwise
    """
    directory = Path(directory)
    archive_extensions = ['.zip', '.tar', '.tar.gz', '.tar.bz2', '.tgz']
    
    archives_found = []
    for ext in archive_extensions:
        archives_found.extend(directory.glob(f"*{ext}"))
    
    if not archives_found:
        print("No archive files found to extract.")
        return True
    
    if not auto_extract:
        print(f"Found {len(archives_found)} archive file(s) but auto-extraction is disabled.")
        return True
    
    print(f"Found {len(archives_found)} archive file(s) to extract:")
    for archive in archives_found:
        print(f"  - {archive.name}")
    
    success = True
    for archive in archives_found:
        if not extract_archive(archive, directory):
            success = False
        elif cleanup_archives:
            try:
                archive.unlink()
                print(f"Removed archive: {archive.name}")
            except Exception as e:
                print(f"Warning: Could not remove archive {archive.name}: {e}")
    
    return success

def download_zenodo_dataset(zenodo_id, dest_dir="./dataset/ctxuxo", auto_extract=True, cleanup_archives=True, reorganize_structure=True):
    """
    Download dataset from Zenodo and extract archives.
    
    Args:
        zenodo_id: Zenodo dataset ID
        dest_dir: Destination directory for the dataset
        auto_extract: Whether to automatically extract archives
        cleanup_archives: Whether to remove archive files after extraction
        reorganize_structure: Whether to reorganize dataset structure
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading dataset from Zenodo ID '{zenodo_id}' to '{dest_dir}'...")
        print("This may take a few minutes depending on your internet connection...")
        
        # Zenodo API endpoint
        api_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        # Get dataset metadata
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error accessing Zenodo record: {response.status_code}")
            return False
        
        record = response.json()
        files = record.get('files', [])
        
        if not files:
            print("No files found in Zenodo record")
            return False
        
        print(f"Found {len(files)} files in Zenodo record")
        
        # Download all files
        for file_info in files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            file_size = file_info['size']
            
            print(f"Downloading {filename} ({file_size / (1024*1024):.1f} MB)...")
            
            # Download file
            file_response = requests.get(download_url, stream=True)
            if file_response.status_code != 200:
                print(f"Error downloading {filename}")
                continue
            
            file_path = dest_path / filename
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {filename}")
        
        print(f"Download complete! Dataset saved in '{dest_dir}'")
        
        # Extract any archives that were downloaded
        print("Checking for archive files to extract...")
        if not find_and_extract_archives(dest_dir, auto_extract, cleanup_archives):
            print("Warning: Some archives could not be extracted properly.")
        
        # Reorganize the dataset structure
        if reorganize_structure:
            print("Reorganizing dataset structure...")
            if not reorganize_dataset_structure(dest_dir):
                print("Warning: Dataset reorganization failed.")
        else:
            print("Dataset reorganization is disabled in configuration.")
        
        # Verify that data.yaml exists
        yaml_path = dest_path / "data.yaml"
        if not yaml_path.exists():
            print(f"Warning: data.yaml not found in {dest_dir}")
            print("The dataset may not be properly structured.")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error downloading dataset from Zenodo: {e}")
        print("Please check:")
        print("1. Internet connection")
        print("2. Zenodo ID is correct")
        print("3. You have sufficient disk space")
        return False

def download_zenodo_with_config(config):
    """
    Download dataset from Zenodo using configuration from config.yaml.
    
    Args:
        config: Configuration dictionary containing dataset settings
        
    Returns:
        bool: True if download successful, False otherwise
    """
    dataset_config = config.get('data', {}).get('dataset', {})
    
    source = dataset_config.get('source', 'huggingface')
    auto_download = dataset_config.get('auto_download', True)
    auto_extract = dataset_config.get('auto_extract', True)
    cleanup_archives = dataset_config.get('cleanup_archives', True)
    reorganize_structure = dataset_config.get('reorganize_structure', True)
    
    if not auto_download:
        print("Auto-download is disabled in configuration.")
        return False
    
    if source == 'zenodo':
        zenodo_id = dataset_config.get('zenodo_id', '1234567')  # Default Zenodo ID
        local_path = dataset_config.get('local_path', './dataset/ctxuxo')
        return download_zenodo_dataset(zenodo_id, local_path, auto_extract, cleanup_archives, reorganize_structure)
    elif source == 'huggingface':
        # TODO: Implement Hugging Face download
        print("Hugging Face download not yet implemented")
        return False
    else:
        print(f"Unknown dataset source: {source}")
        return False

def download_zd_dataset(zenodo_id="1234567", dest_dir="./dataset/ctxuxo"):
    """
    Legacy function for backward compatibility.
    
    Args:
        zenodo_id: Zenodo dataset ID
        dest_dir: Destination directory for the dataset
        
    Returns:
        bool: True if download successful, False otherwise
    """
    return download_zenodo_dataset(zenodo_id, dest_dir)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_zenodo.py <zenodo_id> [dest_dir]")
        print("Example: python get_zenodo.py 1234567 ../ctxuxo")
        sys.exit(1)
    zenodo_id = sys.argv[1]
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else "../ctxuxo"
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"Warning: Directory '{dest_dir}' already exists and is not empty. Files may be overwritten or merged.")
    success = download_zenodo_dataset(zenodo_id, dest_dir)
    sys.exit(0 if success else 1)