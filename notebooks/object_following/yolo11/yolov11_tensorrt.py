import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 加载 TensorRT 引擎
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("yolov8n.trt", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 创建上下文
context = engine.create_execution_context()

# 分配输入输出内存
inputs, outputs, bindings = [], [], []
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))

# 运行推理
def infer(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    stream.synchronize()

# 示例推理输入
image = np.random.randn(1, 3, 640, 640).astype(np.float32)  # 替换为实际图像预处理后的输入
np.copyto(inputs[0][0], image.ravel())
stream = cuda.Stream()

# 执行推理
infer(context, bindings, inputs, outputs, stream)

# 输出结果
output_data = [out[0] for out in outputs]
print("推理结果:", output_data)