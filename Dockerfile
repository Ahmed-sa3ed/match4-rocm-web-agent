FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
ARG VLLM_REF
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf-cache
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ninja-build libnuma-dev && rm -rf /var/lib/apt/lists/*
COPY scripts/build_vllm.sh /app/scripts/build_vllm.sh
RUN VLLM_REF="${VLLM_REF}" bash /app/scripts/build_vllm.sh
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -c /tmp/match4-torch-constraints.txt -r requirements.txt && scrapling install
COPY . /app
RUN python3 -c "import torch; assert torch.version.hip; print(torch.__version__)"
HEALTHCHECK --interval=10s --timeout=5s --start-period=570s CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()"
CMD ["python3", "/app/serve.py"]
