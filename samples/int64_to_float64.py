import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, TensorProto

# Define the ONNX model
input_tensor = helper.make_tensor_value_info('input', TensorProto.INT64, [None, 1])
output_tensor = helper.make_tensor_value_info('output', TensorProto.DOUBLE, [None, 1])

cast_node = helper.make_node(
    'Cast',
    inputs=['input'],
    outputs=['output'],
    to=TensorProto.DOUBLE
)

graph_def = helper.make_graph(
    [cast_node],
    'Int64ToFloat64Model',
    [input_tensor],
    [output_tensor]
)

model_def = helper.make_model(graph_def, producer_name='onnx-example', opset_imports=[helper.make_operatorsetid("", 21)])
onnx.save(model_def, 'samples/int64_to_float64.onnx')

# Load and test the model
session = ort.InferenceSession('samples/int64_to_float64.onnx', providers=['CPUExecutionProvider'])
input_data = np.array([[1], [2], [3]], dtype=np.int64)
outputs = session.run(None, {'input': input_data})

print(outputs[0])  # Should be float64 values
