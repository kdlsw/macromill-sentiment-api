#!/usr/bin/env python3
"""Create a checkpoint zip of the current project state."""

import zipfile
import os
from pathlib import Path
from datetime import datetime

# Directories to exclude entirely
EXCLUDE_DIRS = {'.venv', '.git', 'checkpoints', '__pycache__', 'cuda-packages'}

def create_checkpoint():
    root = Path('/home/ubun/macromill')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = root / 'checkpoints' / f'macromill_clean_{timestamp}.zip'
    
    print(f"Creating checkpoint: {output_path}")
    
    files_to_zip = []
    
    for item in root.rglob('*'):
        # Skip excluded directories entirely
        if any(excluded in item.parts for excluded in EXCLUDE_DIRS):
            continue
        
        if item.is_file():
            arcname = item.relative_to(root)
            files_to_zip.append((item, arcname))
    
    print(f"Adding {len(files_to_zip)} files...")
    
    # Create zip
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for item, arcname in files_to_zip:
            zf.write(item, arcname)
    
    print(f"Checkpoint saved to: {output_path}")
    
    # Show size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

if __name__ == '__main__':
    create_checkpoint()
