from __future__ import annotations
import argparse, collections, hashlib, json, random, re, unicodedata
from pathlib import Path
from PIL import Image


def normalized(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn')


def prepare(root: Path, output: Path, seed=42):
    rows, hashes = [], {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        name = normalized(folder.name)
        label = 'cube' if 'lap' in name or name == 'cube' else 'cylinder' if 'tru' in name or name == 'cylinder' else None
        if label is None: continue
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
            with Image.open(path) as image: image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'Exact duplicate: {path} and {hashes[digest]}. Remove from the chosen dataset, not automatically.')
            hashes[digest] = str(path)
            m = re.search(r'(\d{8})_(\d{2})_(\d{2})_(\d{2})', path.name)
            group = f'{m[1]}_{m[2]}_{m[3]}_{int(m[4])//5}' if m else path.stem
            rows.append(dict(path=str(path.resolve()), label=label, group=group, sha256=digest))
    if set(r['label'] for r in rows) != {'cube','cylinder'}:
        raise ValueError('Need both cube and cylinder image directories.')
    rng = random.Random(seed)
    for label in ['cube','cylinder']:
        groups = sorted({r['group'] for r in rows if r['label'] == label})
        if len(groups) < 3: raise ValueError('Need at least three capture groups per class for separate train/val/test.')
        rng.shuffle(groups)
        n = max(1, round(len(groups)*0.15))
        assignments = {g: 'test' if i < n else 'val' if i < 2*n else 'train' for i,g in enumerate(groups)}
        for row in rows:
            if row['label'] == label: row['split'] = assignments[row['group']]
    result = {'seed': seed, 'source_root': str(root.resolve()), 'total_original_images': len(rows),
              'counts': dict(collections.Counter(f"{r['split']}/{r['label']}" for r in rows)),
              'grouping': 'filename capture-time buckets of 5 seconds, separated within each class',
              'limitations': 'Adjacent frames and a single background are not independent physical-product sessions. Validate again on the conveyor.',
              'images': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'models/dataset_manifest.json')
    a=p.parse_args()
    result=prepare(a.root,a.output)
    print(json.dumps({k:v for k,v in result.items() if k != 'images'},ensure_ascii=True,indent=2))
