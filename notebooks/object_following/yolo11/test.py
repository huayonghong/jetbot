import tensorrt as trt
import numpy as np


# 加载 TensorRT 引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("yolo11n.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# 创建上下文
context = engine.create_execution_context()