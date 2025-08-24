{
    files = {
        "build/.objs/infiniccl/linux/x86_64/release/src/infiniccl/infiniccl.cc.o",
        "build/linux/x86_64/release/libinfini-utils.a",
        "build/linux/x86_64/release/libinfinirt-cpu.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-shared",
            "-fPIC",
            "-m64",
            "-Lbuild/linux/x86_64/release",
            "-s",
            "-linfinirt",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}