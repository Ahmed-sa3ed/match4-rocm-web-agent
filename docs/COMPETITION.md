# Competition readiness

## Sources reviewed

The supplied AMD AI League Match 1 & 2 brief (shared infrastructure and submission section), Match 3 brief and starter-kit structure, newsletter dated 8 October, and original notebook checkpoint were reviewed. The checkpoint used Qwen/Qwen3-4B-Instruct-2507 through Transformers on an AMD MI300X VF.

The shared brief mandates final base `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, preservation of its layers, code under `/app`, an image no larger than 60 GiB uncompressed, and 1–48 GiB measured VRAM with its stated upper tolerance. It describes submission of a publicly pullable registry image. Per that brief, do not publish the final image reference in the public source repository.

Earlier matches specify 10-minute startup and 30-second task invocations. Match 3 explicitly recommends holding the model in a resident process between CLI calls. This project adopts that architecture but has not measured those budgets on the AMD environment.

## Match 4 contract is unresolved

The public Match 4 details link returned to the event page when checked. The supplied documents do not establish its exact inputs, output JSON schema, navigation scope or timing limits. The OCR and document-indexing interfaces from earlier matches are not assumed to apply. The project's `app.py --url --query --query-id` interface is provisional pending the actual contract.

## Before graded submission

1. Obtain the official Match 4 invocation/output specification and adapt the boundary if needed.
2. Build the checkpoint Transformers server in the mandated base; confirm ROCm torch survives installation and the model cache fits.
3. Run real Qwen queries on the evaluation-class GPU, including static HTML, JavaScript and PDF cases.
4. Measure cold startup, warm-query latency, GPU peak and uncompressed image size. The current code has no hard whole-query deadline.
5. Record the exact PyTorch, Transformers and model revisions for reproducibility.
6. If the shared registry submission rules apply, publish the tested image and supply its reference through the organizer's submission form. Keep that reference out of this repository.

## Completed checks

Backend, feature, adaptive, conflict and PDF self-checks passed on development Windows Python 3.12. The 33-case configuration passes a dry run. These checks use mocked inference and retrieval where needed. The user validated a local Scrapling Chromium render with the notebook no-zygote setting. Docker and live Qwen accuracy with the updated agent remain unverified.

References: [event](https://lablab.ai/ai-hackathons/amd-lablab-ai-academy-challenge), and the supplied briefs.
