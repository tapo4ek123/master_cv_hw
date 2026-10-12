"""Запустить perf_analyzer для нескольких уровней concurrency, сохранить CSV и логи."""
import argparse
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen


def server_stats():
    with urlopen('http://localhost:8000/v2/models/resnet18/stats', timeout=5) as response:
        return json.load(response)['model_stats'][0]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('results/perf_baseline'))
parser.add_argument('--url', default='localhost:8001')
parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 2, 4, 8, 16])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
statistics = []
for concurrency in args.concurrency:
    before = server_stats()
    command = ['perf_analyzer', '-m', 'resnet18', '-i', 'grpc', '-u', args.url,
               '-b', '1', '--concurrency-range', str(concurrency),
               '--measurement-interval', '5000', '--stability-percentage', '20',
               '--max-trials', '20',
               '-f', str(args.output / f'concurrency_{concurrency}.csv')]
    print(' '.join(command), flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.output / f'concurrency_{concurrency}.log').write_text(result.stdout, encoding='utf8')
    print(result.stdout, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    after = server_stats()
    images = int(after['inference_count']) - int(before['inference_count'])
    executions = int(after['execution_count']) - int(before['execution_count'])
    statistics.append({'concurrency': concurrency, 'images': images, 'executions': executions,
                       'average_execution_batch': images / executions if executions else None})
    (args.output / 'server_stats.json').write_text(json.dumps(statistics, indent=2), encoding='utf8')
