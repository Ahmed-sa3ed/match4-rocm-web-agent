FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf-cache
WORKDIR /app
RUN python3 -c 'import torch; print("torch==" + torch.__version__)' > /tmp/match4-torch-constraints.txt
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -c /tmp/match4-torch-constraints.txt -r requirements.txt && scrapling install
COPY . /app
RUN python3 -c "import torch, transformers; assert torch.version.hip; print(torch.__version__, transformers.__version__)"
HEALTHCHECK --interval=10s --timeout=5s --start-period=570s CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=3).raise_for_status()"
CMD ["python3", "/app/serve.py"]
