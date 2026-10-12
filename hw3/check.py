"""Проверить HTTP/gRPC, результаты локального ResNet-18 и метрики под нагрузкой."""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from torchvision.models import resnet18

from client import infer, make_client
from preprocess import prepare_image


def read_metrics():
    with urlopen('http://localhost:8002/metrics', timeout=5) as response:
        text = response.read().decode()
    values = {'counter': 0.0, 'gauge': 0.0}
    for line in text.splitlines():
        if line.startswith('resnet18_processing_seconds_total{'):
            values['counter'] += float(line.split()[-1])
        elif line.startswith('resnet18_requests_in_progress{'):
            values['gauge'] += float(line.split()[-1])
    if 'resnet18_processing_seconds_total{' not in text:
        raise AssertionError('Кастомные метрики не найдены')
    return values, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/correctness.json'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    batch = np.stack([prepare_image(path) for path in args.images])
    local = resnet18(weights=None)
    local.load_state_dict(torch.load('/models/resnet18/1/model.pth', map_location='cpu', weights_only=True))
    local.to(args.device).eval()
    with torch.inference_mode():
        # Сравниваем одинаковые batch sizes: CUDA может выбирать разные kernels.
        expected = np.concatenate([
            local(torch.from_numpy(image[None]).to(args.device)).cpu().numpy() for image in batch])
    results = {'images': [str(path) for path in args.images],
               'reference_same_batch_size': True, 'protocols': {}}
    for protocol, url in [('http', 'localhost:8000'), ('grpc', 'localhost:8001')]:
        api, client = make_client(protocol, url)
        actual = np.concatenate([infer(client, api, image[None]) for image in batch])
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        results['protocols'][protocol] = {
            'max_abs_diff': float(np.max(np.abs(actual - expected))),
            'top1_matches': bool(np.array_equal(actual.argmax(1), expected.argmax(1))),
        }
        if args.batch:
            together = infer(client, api, batch)
            with torch.inference_mode():
                expected_batch = local(torch.from_numpy(batch).to(args.device)).cpu().numpy()
            np.testing.assert_allclose(together, expected_batch, rtol=1e-3, atol=1e-4)
            assert np.array_equal(together.argmax(1), actual.argmax(1)), 'Batch изменил top-1'
            results['protocols'][protocol]['batch_reference_max_abs_diff'] = float(np.max(np.abs(together - expected_batch)))
            results['protocols'][protocol]['different_batch_sizes_max_abs_diff'] = float(np.max(np.abs(together - actual)))
        client.close()
    before, _ = read_metrics()
    assert before['gauge'] == 0, before

    def worker():
        api, client = make_client('http', 'localhost:8000')
        try:
            for _ in range(50):
                output = infer(client, api, batch[:1])
                assert np.isfinite(output).all()
                assert output.argmax(1)[0] == expected.argmax(1)[0]
        finally:
            client.close()

    samples = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(8)]
        while any(not future.done() for future in futures):
            values, _ = read_metrics()
            samples.append(values)
            time.sleep(0.005)
        for future in futures:
            future.result()
    after, text = read_metrics()
    assert after['counter'] > before['counter'], (before, after)
    assert after['gauge'] == 0, after
    assert samples and max(row['gauge'] for row in samples) > 0, 'GAUGE не замечен под нагрузкой'
    assert all(row['gauge'] >= 0 for row in samples), samples
    counters = [before['counter']] + [row['counter'] for row in samples] + [after['counter']]
    assert all(a <= b for a, b in zip(counters, counters[1:])), 'COUNTER уменьшился'
    results['metrics'] = {'before': before, 'after': after,
                         'maximum_gauge': max(row['gauge'] for row in samples),
                         'load_requests': 400, 'samples': len(samples)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf8')
    args.output.with_suffix('.metrics.txt').write_text(text, encoding='utf8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
