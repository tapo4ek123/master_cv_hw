"""ResNet-18, настоящий batching и две метрики Triton."""
import os
import time
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torchvision.models import resnet18


class TritonPythonModel:
    def initialize(self, args):
        torch.set_num_threads(1)
        self.device = torch.device(
            f"cuda:{args['model_instance_device_id']}"
            if args['model_instance_kind'] == 'GPU' else 'cpu')
        self.model = resnet18(weights=None)
        weights = Path(__file__).parent / 'model.pth'
        self.model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.model.to(self.device).eval()
        # Прогрев не входит в пользовательские метрики.
        with torch.inference_mode():
            for _ in range(3):
                self.model(torch.zeros(1, 3, 224, 224, device=self.device))
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        labels = {'model': args['model_name'], 'version': args['model_version'],
                  'instance': args.get('model_instance_name', str(os.getpid()))}
        self.time_family = pb_utils.MetricFamily(
            name='resnet18_processing_seconds_total',
            description='Sum of processing durations of individual requests; excludes queue time',
            kind=pb_utils.MetricFamily.COUNTER)
        self.active_family = pb_utils.MetricFamily(
            name='resnet18_requests_in_progress',
            description='Requests currently executing, excluding requests in the Triton queue',
            kind=pb_utils.MetricFamily.GAUGE)
        self.processing_time = self.time_family.Metric(labels=labels)
        self.active = self.active_family.Metric(labels=labels)

    def execute(self, requests):
        count = len(requests)
        self.active.increment(count)
        started = time.perf_counter()
        try:
            arrays = [pb_utils.get_input_tensor_by_name(r, 'IMAGE').as_numpy() for r in requests]
            sizes = [len(array) for array in arrays]
            batch = torch.from_numpy(np.concatenate(arrays, axis=0)).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch).float().cpu().numpy()
            # Перенос .cpu() дожидается GPU: время ниже включает завершённый forward.
            responses, offset = [], 0
            for size in sizes:
                output = pb_utils.Tensor('LOGITS', logits[offset:offset + size].copy())
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
                offset += size
            return responses
        except Exception as error:
            return [pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))) for _ in requests]
        finally:
            # Каждый запрос провёл в общем batch одинаковое время. Считаем сумму
            # длительностей запросов, а не GPU seconds и не число изображений.
            self.processing_time.increment((time.perf_counter() - started) * count)
            # В Python backend 23.12 есть increment/set/value, но нет decrement.
            self.active.increment(-count)

    def finalize(self):
        if hasattr(self, 'processing_time'):
            del self.processing_time, self.active
            del self.time_family, self.active_family
