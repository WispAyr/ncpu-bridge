"""
Hailo-8 Deployment Status & Benchmark Results
===============================================
ONNX models running on Pi ARM CPU. Hailo-8 hardware available but
needs DFC (Dataflow Compiler) to compile ONNX → HEF format.

Pi: 192.168.195.238 (pi/pi)
Hailo-8: firmware 4.23.0, HailoRT 4.23.0
Models: /home/pi/ncpu-hailo/onnx_models/ (15 ONNX files)
Venv: /home/pi/ncpu-hailo/venv/ (onnxruntime 1.24.4)

BENCHMARK RESULTS (Pi ARM Cortex-A76, onnxruntime):
=====================================================
compare:       8.1 µs → 123,568 ops/sec  (55x faster than PyTorch)
carry_combine: 10.4 µs →  96,257 ops/sec  (43x)
divide:        10.4 µs →  95,923 ops/sec  (43x)
arithmetic:    11.4 µs →  87,805 ops/sec  (39x)
logical:       12.2 µs →  82,067 ops/sec  (37x)
exp:           29.2 µs →  34,243 ops/sec
log:           29.2 µs →  34,285 ops/sec
sqrt:          95.7 µs →  10,444 ops/sec
multiply:     224.9 µs →   4,446 ops/sec
sincos:       242.4 µs →   4,125 ops/sec
atan2:        597.7 µs →   1,673 ops/sec
rol:        1,537.7 µs →     650 ops/sec
asr:        1,692.6 µs →     591 ops/sec
lsr:        3,366.7 µs →     297 ops/sec
lsl:        3,396.8 µs →     294 ops/sec

TOTAL: 576,668 ops/sec (ARM CPU via ONNX Runtime)

NEXT: Hailo-8 (26 TOPS) should do 10-100x more.

TO GET HAILO-8 ACCELERATION:
===============================
1. Register at https://hailo.ai/developer-zone/
2. Download Hailo Dataflow Compiler (DFC) — x86 Linux only
3. Install DFC on an x86 Linux machine (or Docker):
   pip install hailo_dataflow_compiler-X.X.X-pyX-none-linux_x86_64.whl
4. For each model, compile:
   hailo parser arithmetic.onnx
   hailo optimize arithmetic_hailo_model.hn
   hailo compiler arithmetic_hailo_model.hn --hw-arch hailo8
5. SCP the resulting .hef files to Pi:
   scp *.hef pi@192.168.195.238:/home/pi/ncpu-hailo/hef_models/
6. Run inference via HailoRT Python API:
   from hailo_platform import HEF, VDevice, InferVStreams
   hef = HEF('arithmetic.hef')
   ...

ALTERNATIVE: Hailo Model Zoo Docker (includes DFC):
   docker pull hailo-ai/hailo_model_zoo:2.13.0
   docker run -v ./onnx:/data hailo-ai/hailo_model_zoo:2.13.0 \\
     hailomz compile --onnx /data/arithmetic.onnx --hw-arch hailo8
"""

# This file documents deployment status — not executable code
