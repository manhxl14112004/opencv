"""Resolve image paths relative to their manifest so datasets can be moved."""
import json
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    for row in data['images']:
        image = Path(row['path'])
        row['path'] = str(image if image.is_absolute() else (path.parent / image).resolve())
    return data
