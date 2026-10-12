ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.12-py3
ARG TRITON_SDK_IMAGE=nvcr.io/nvidia/tritonserver:23.12-py3-sdk
FROM ${TRITON_SDK_IMAGE} AS sdk
FROM ${TRITON_IMAGE}
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
WORKDIR /work
COPY --from=sdk /workspace/install/bin/perf_analyzer /usr/local/bin/perf_analyzer
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends libb64-0d wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install --no-cache-dir -r requirements.txt
COPY model_repository /models
COPY assets /work/assets
COPY prepare.py /work/prepare.py
# prepare.py работает с относительной к скрипту папкой: подготовим /models.
RUN mkdir -p /work/model_repository/resnet18/1 \
    && cp -a /models/resnet18/1/. /work/model_repository/resnet18/1/ \
    && python3 prepare.py \
    && cp /work/model_repository/resnet18/1/model.pth /models/resnet18/1/model.pth \
    && rm -rf /work/model_repository /root/.cache/torch
CMD ["tritonserver", "--model-repository=/models", "--strict-model-config=true"]
