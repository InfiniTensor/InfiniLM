{
    files = {
        "build/.objs/infinirt/linux/x86_64/release/src/infinirt/infinirt.cc.o",
        "build/linux/x86_64/release/libinfini-utils.a",
        "build/linux/x86_64/release/libinfinirt-cpu.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-shared",
            "-m64",
            "-fPIC",
            "-Lbuild/linux/x86_64/release",
            "-s",
            "-linfinirt-cpu",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}