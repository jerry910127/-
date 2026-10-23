"""水果 CNN：訓練、驗證、最佳權重及獨立測試。"""
import argparse
import copy
import csv
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from cnn import CNN
from dataset import CLASSES, FruitDataset, prepare_data
from eda import plot_eda


def run(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total, correct, loss_sum, actual, predicted, confidence = 0, 0, 0., [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            conf, p = logits.softmax(1).max(1)
            total += len(y)
            loss_sum += loss.item()*len(y)
            correct += (p == y).sum().item()
            actual.extend(y.cpu().tolist())
            predicted.extend(p.cpu().tolist())
            confidence.extend(conf.detach().cpu().tolist())
    return loss_sum/total, correct/total, actual, predicted, confidence


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, default=root/'fruit.zip')
    parser.add_argument('--out-dir', type=Path, default=root/'result')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--split', choices=['group', 'original'], default='group',
                        help='group：按原始照片分組切分（預設，避免洩漏）；original：沿用資料集原本的 train/test')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 16 or args.lr <= 0:
        parser.error('epochs/batch-size/lr 必須為正，size 至少 16')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    # GPU 上部分運算（如 AdaptiveAvgPool 反向傳播）沒有確定性版本：改成只警告，不中斷訓練
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print('Reading fruit ZIP and checking exact duplicates...', flush=True)
    images, records, metadata = prepare_data(args.archive, args.size, split=args.split)
    print(f"Split={args.split}: {metadata['source_photos']:,} source photos, "
          f"test leakage {metadata['leak_original_split']:.1%} (original split) -> {metadata['leak_this_split']:.1%} (this run)", flush=True)
    (out/'data_audit.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (out/'split_manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'label', 'split', 'sha256', 'source'])
        writer.writeheader(); writer.writerows(records)
    print('Plotting EDA figures...', flush=True)
    plot_eda(args.archive, out, metadata)
    loaders = {s: DataLoader(FruitDataset(images, records, s), batch_size=args.batch_size,
                  shuffle=s == 'train', num_workers=0,
                  generator=torch.Generator().manual_seed(42)) for s in ('train', 'val', 'test')}
    print({s: len(loader.dataset) for s, loader in loaders.items()}, flush=True)
    model = CNN(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best_loss, best_state, best_epoch, stale = [], float('inf'), None, 0, 0
    # 只以驗證 loss 挑選權重，測試集不參與選模型。
    for epoch in range(1, args.epochs+1):
        tr = run(model, loaders['train'], device, optimizer)
        va = run(model, loaders['val'], device)
        history.append([epoch, tr[0], va[0], tr[1], va[1]])
        print(f'Epoch {epoch}: train_loss={tr[0]:.4f} val_loss={va[0]:.4f} train_acc={tr[1]:.4f} val_acc={va[1]:.4f}', flush=True)
        if va[0] < best_loss:
            best_loss, best_epoch, stale = va[0], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(best_state)
    loss, acc, actual, predicted, confidence = run(model, loaders['test'], device)
    cm = np.zeros((6, 6), dtype=int)
    for y, p in zip(actual, predicted):
        cm[y, p] += 1
    rows = []
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i]); precision = tp/max(1, int(cm[:, i].sum()))
        recall = tp/max(1, int(cm[i].sum()))
        rows.append({'class': name, 'precision': precision, 'recall': recall,
                     'f1': 2*precision*recall/max(1e-12, precision+recall), 'support': int(cm[i].sum())})
    majority = max(range(6), key=lambda i: metadata['counts']['train'][CLASSES[i]])
    result = {'task': 'Fruit type and visual freshness classification', 'seed': 42,
              'device': str(device), 'epochs_run': len(history), 'best_epoch': best_epoch,
              'test_loss': loss, 'test_accuracy': acc,
              'macro_f1': float(np.mean([r['f1'] for r in rows])),
              'majority_baseline_accuracy': actual.count(majority)/len(actual),
              'per_class': rows, 'confusion_matrix': cm.tolist(), 'metadata': metadata,
              'params': sum(p.numel() for p in model.parameters()), 'image_size': args.size,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'os': platform.platform(), 'cpu': platform.processor(),
                              'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None},
              'elapsed_seconds': time.time()-start}
    (out/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (out/'loss.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']); w.writerows(history)
    with (out/'test_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['file', 'actual', 'predicted', 'confidence'])
        w.writerows((records[i]['file'], CLASSES[y], CLASSES[p], f'{c:.4f}')
                    for i, y, p, c in zip(loaders['test'].dataset.ids, actual, predicted, confidence))
    torch.save({'model_state_dict': best_state, 'classes': CLASSES,
                'image_size': args.size, 'normalization': {'mean': .5, 'std': .5},
                'best_epoch': best_epoch}, out/'best_cnn.pth')
    from plot_results import plot_results
    plot_results(out, args.archive)
    print(json.dumps({k: result[k] for k in ('test_accuracy', 'macro_f1', 'best_epoch', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
