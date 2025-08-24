{
    depfiles_format = "gcc",
    depfiles = "strorage.o: src/tensor/strorage.cpp src/tensor/../allocator.hpp  include/infinicore_infer.h include/infinicore_infer/models/qwen3.h  /home/hootandy/.infini/include/infiniccl.h  /home/hootandy/.infini/include/infinirt.h  /home/hootandy/.infini/include/infinicore.h  /home/hootandy/.infini/include/infiniop.h  /home/hootandy/.infini/include/infiniop/handle.h  /home/hootandy/.infini/include/infiniop/../infinicore.h  /home/hootandy/.infini/include/infiniop/ops/add.h  /home/hootandy/.infini/include/infiniop/ops/../operator_descriptor.h  /home/hootandy/.infini/include/infiniop/ops/../handle.h  /home/hootandy/.infini/include/infiniop/ops/../tensor_descriptor.h  /home/hootandy/.infini/include/infiniop/ops/../../infinicore.h  /home/hootandy/.infini/include/infiniop/ops/attention.h  /home/hootandy/.infini/include/infiniop/ops/gemm.h  /home/hootandy/.infini/include/infiniop/ops/swiglu.h  /home/hootandy/.infini/include/infiniop/ops/causal_softmax.h  /home/hootandy/.infini/include/infiniop/ops/clip.h  /home/hootandy/.infini/include/infiniop/ops/conv.h  /home/hootandy/.infini/include/infiniop/ops/gemm.h  /home/hootandy/.infini/include/infiniop/ops/linear.h  /home/hootandy/.infini/include/infiniop/ops/linear_backwards.h  /home/hootandy/.infini/include/infiniop/ops/mul.h  /home/hootandy/.infini/include/infiniop/ops/random_sample.h  /home/hootandy/.infini/include/infiniop/ops/rearrange.h  /home/hootandy/.infini/include/infiniop/ops/relu.h  /home/hootandy/.infini/include/infiniop/ops/rms_norm.h  /home/hootandy/.infini/include/infiniop/ops/rope.h  /home/hootandy/.infini/include/infiniop/ops/sub.h  /home/hootandy/.infini/include/infiniop/ops/swiglu.h  /home/hootandy/.infini/include/infiniop/tensor_descriptor.h  /home/hootandy/.infini/include/infinirt.h src/tensor/../tensor.hpp  src/tensor/../allocator.hpp src/tensor/../utils.hpp  /home/hootandy/.infini/include/infinicore.h\
",
    files = {
        "src/tensor/strorage.cpp"
    },
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fPIC",
            "-Wall",
            "-Werror",
            "-std=c++17",
            "-Iinclude",
            "-I/home/hootandy/.infini/include"
        }
    }
}