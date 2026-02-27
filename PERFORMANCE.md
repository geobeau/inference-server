This is a log of performance improvement

Unfortunately, perf analyser is finicky so it's harder to get stable measurements.


# Pajamax x Compio





The current baseline is:
* 2 server core + 4 onnx intra thread: 
    * 60kQPS -> 180% cpu
    * 120k is overloading
    * 82kQPS -> 240% cpu

# Avoid inputs name string and hashmap overhead

At the moment, we do a lot of:
- creating hashmap to store inputs
- cloning the String (the input names) everywhere

So intead, we forward everything as a slice ordered in the same order as the inputs of the models.
This prevents uneeded heap allocation and string comparisons.

# Avoid tensor copies in the GRPC handled

We copy each request into a Tensor structure but eventually it is copied back to the supertensor.
We can try to carry the vec<u8> of the request directly to the supertensor.

Note: not done yet

# Using cuda pinned memory

This will help not having to copy the memory within onnx to move to it to the GPU. The GPU can
access the cuda memory itself if it's flagged as cuda pinned


