import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto

# Define dimensions
M, K, N = 1024, 1024, 1024


print(ort.get_available_providers())

# Build model with MatMul node
input_A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
input_B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [K, N])
output_C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [M, N])

matmul_node = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])

graph_def = helper.make_graph([matmul_node], 'MatMulModel', [input_A, input_B], [output_C])
model_def = helper.make_model(graph_def, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save(model_def, 'samples/matmul.onnx')

# Inference
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Use GPU execution provider if available
session = ort.InferenceSession('samples/matmul.onnx', providers=['CPUExecutionProvider', 'OpenVINOExecutionProvider'])
C = session.run(None, {'A': A, 'B': B})[0]

print(C.shape)
