"""Отправить картинки через HTTP/gRPC и вывести top-5 ImageNet-классов."""
import argparse
import json
from pathlib import Path

import numpy as np
from preprocess import prepare_image


def make_client(protocol, url):
    if protocol == 'http':
        import tritonclient.http as api
    else:
        import tritonclient.grpc as api
    return api, api.InferenceServerClient(url=url)


def infer(client, api, batch):
    tensor = api.InferInput('IMAGE', batch.shape, 'FP32')
    tensor.set_data_from_numpy(batch)
    result = client.infer('resnet18', [tensor], outputs=[api.InferRequestedOutput('LOGITS')])
    return result.as_numpy('LOGITS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--protocol', choices=['http', 'grpc'], default='http')
    parser.add_argument('--url', help='По умолчанию localhost:8000 для HTTP, :8001 для gRPC')
    parser.add_argument('--batch', action='store_true', help='Отправить все картинки одним запросом')
    parser.add_argument('--output', type=Path, default=Path('results/predictions.json'))
    args = parser.parse_args()
    api, client = make_client(args.protocol, args.url or ('localhost:8000' if args.protocol == 'http' else 'localhost:8001'))
    if not client.is_server_ready() or not client.is_model_ready('resnet18'):
        raise RuntimeError('Triton/ResNet-18 ещё не готов')
    names = json.loads((Path(__file__).parent / 'assets/classes.json').read_text())
    batch = np.stack([prepare_image(path) for path in args.images])
    if args.batch:
        logits = infer(client, api, batch)
    else:
        logits = np.concatenate([infer(client, api, image[None]) for image in batch])
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    records = []
    for path, values in zip(args.images, probabilities):
        top = np.argsort(values)[-5:][::-1]
        record = {'image': str(path), 'top5': [{'class_id': int(i), 'class': names[i],
                                             'probability': float(values[i])} for i in top]}
        records.append(record)
        print(f'{path}: ' + ', '.join(f"{names[i]} ({values[i]:.1%})" for i in top))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'protocol': args.protocol, 'results': records}, indent=2), encoding='utf8')
    client.close()


if __name__ == '__main__':
    main()
