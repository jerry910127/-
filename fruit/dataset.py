"""直接讀 ZIP，固定分割並快取縮圖；不更動原壓縮檔。"""
import hashlib
import io
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torch

CLASSES = ['freshapples', 'freshbanana', 'freshoranges',
           'rottenapples', 'rottenbanana', 'rottenoranges']
# 資料集作者已離線做過增強：同一張原始照片另有旋轉、平移、椒鹽雜訊、垂直翻轉等版本，檔名以這些前綴開頭。
AUG_PREFIX = re.compile(r'^(?:rotated_by_-?\d+_|translation_|saltandpepper_|vertical_flip_|horizontal_flip_)+')
TEST_RATIO, VAL_RATIO = .2, .15  # 分組切分：20% 原始照片作測試，其餘的 15% 作驗證


def source_id(name):
    """去掉增強前綴，取得原始照片名稱；同一張照片的所有變體會得到相同的 id。"""
    return AUG_PREFIX.sub('', Path(name).stem)


def augmentation(name):
    """檔名對應的離線增強種類（rotated / translation / ...），沒有前綴就是 original。"""
    m = AUG_PREFIX.match(Path(name).name)
    return re.sub(r'_by_-?\d+', '', m.group(0)).strip('_') if m else 'original'


def leak_ratio(records):
    """測試圖中，原始照片也出現在 train / val 的比例。"""
    seen = {(r['label'], r['source']) for r in records if r['split'] != 'test'}
    test = [r for r in records if r['split'] == 'test']
    return sum((r['label'], r['source']) in seen for r in test) / max(1, len(test))


def prepare_data(archive, size=64, seed=42, split='group'):
    rng = random.Random(seed)
    records, images, seen = [], [], {}
    skipped = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist()
                       if n.startswith(('dataset/train/', 'dataset/test/'))
                       and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'})
        if not names:
            raise ValueError('ZIP 缺少 dataset/train 或 dataset/test 影像')
        for name in names:
            parts = name.split('/')
            if parts[2] not in CLASSES:
                raise ValueError(f'未知類別: {name}')
            raw = z.read(name)
            # 以原始檔案內容雜湊去除完全相同的影像（含跨組重複）。
            digest = hashlib.sha256(raw).hexdigest()
            if digest in seen:
                if seen[digest] != parts[2]:
                    raise ValueError('重複影像有矛盾標籤')
                continue
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    a = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((size, size)), dtype=np.uint8).copy()
            except (OSError, ValueError) as exc:
                skipped.append({'file': name, 'reason': str(exc)})
                continue
            seen[digest] = parts[2]
            images.append(a)
            records.append({'file': name, 'label': CLASSES.index(parts[2]),
                            'split': parts[1], 'sha256': digest, 'source': source_id(name)})
    # 先記錄資料集原本 train / test 切分的洩漏程度，作為對照。
    original_leak = leak_ratio(records)
    if split == 'original':
        # 保留原 test；只由 train 逐類抽 15% 作驗證。
        for label in range(len(CLASSES)):
            ids = [i for i, r in enumerate(records) if r['split'] == 'train' and r['label'] == label]
            rng.shuffle(ids)
            if len(ids) < 2:
                raise ValueError('每個類別至少需兩張訓練圖')
            for i in ids[:max(1, round(len(ids)*.15))]:
                records[i]['split'] = 'val'
    elif split == 'group':
        # 以原始照片為單位逐類重新切分：同一張照片的所有變體只會在同一組，測試集都是模型沒看過的照片。
        for label in range(len(CLASSES)):
            sources = sorted({r['source'] for r in records if r['label'] == label})
            if len(sources) < 3:
                raise ValueError('每個類別至少需三張原始照片')
            rng.shuffle(sources)
            n_test = max(1, round(len(sources)*TEST_RATIO))
            n_val = max(1, round((len(sources)-n_test)*VAL_RATIO))
            group = {s: 'test' for s in sources[:n_test]}
            group.update({s: 'val' for s in sources[n_test:n_test+n_val]})
            for r in records:
                if r['label'] == label:
                    r['split'] = group.get(r['source'], 'train')
    else:
        raise ValueError(f'未知的切分方式: {split}')
    counts = {s: dict(Counter(CLASSES[r['label']] for r in records if r['split'] == s)) for s in ('train', 'val', 'test')}
    if any(len(counts[s]) != len(CLASSES) for s in counts):
        raise ValueError('分割後各組都必須包含六類')
    variants = Counter(Counter(r['source'] for r in records).values())
    augment = Counter(augmentation(r['file']) for r in records)
    return np.stack(images), records, {'counts': counts, 'skipped': skipped,
        'canonical_entries': len(names), 'unique_images': len(records),
        'duplicates_removed': len(names)-len(records)-len(skipped), 'image_size': size,
        'split_method': split,
        'source_photos': len({(r['label'], r['source']) for r in records}),
        'variants_per_photo': {str(k): v for k, v in sorted(variants.items())},
        'augmentation_types': dict(augment),
        'leak_original_split': original_leak, 'leak_this_split': leak_ratio(records),
        'limitation': 'Augmented variants are grouped by file-name prefix; photos taken in the same session may still look alike.'}


class FruitDataset(Dataset):
    def __init__(self, images, records, split):
        self.images, self.records, self.split = images, records, split
        self.ids = [i for i, r in enumerate(records) if r['split'] == split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        x = torch.from_numpy(self.images[i].copy()).permute(2, 0, 1).float()/255
        # 資料增強只套用訓練集；驗證與測試保持固定。
        if self.split == 'train' and torch.rand(()) < .5:
            x = x.flip(2)
        return (x-.5)/.5, self.records[i]['label']
