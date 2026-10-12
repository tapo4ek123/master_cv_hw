"""Скачать готовые веса ResNet-18 и сохранить список классов."""
import json
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

root = Path(__file__).resolve().parent
weights = ResNet18_Weights.IMAGENET1K_V1
path = root / 'model_repository/resnet18/1/model.pth'
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    state = weights.get_state_dict(progress=True, check_hash=True)
    torch.save(state, path)
(root / 'assets').mkdir(exist_ok=True)
(root / 'assets/classes.json').write_text(json.dumps(weights.meta['categories'], indent=2), encoding='utf8')
print(f'Weights: {path}', flush=True)
