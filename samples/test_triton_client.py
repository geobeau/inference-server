import argparse
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc

FLAGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )

    FLAGS = parser.parse_args()

    model_name = "Int64ToFloat64Model"
    model_version = ""
    batch_size = 1

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)


    # Infer
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = "my request id"

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = "input"
    input.datatype = "INT64"
    input.shape.extend([1, 1])
    input.contents.int_contents[:] = [1,1]
    request.inputs.extend([input])

    # doesn't work :(
    response = grpc_stub.ModelInfer(request)
    print("model infer:\n{}".format(response))


