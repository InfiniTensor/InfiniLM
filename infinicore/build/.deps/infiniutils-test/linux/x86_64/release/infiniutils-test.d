{
    files = {
        "build/.objs/infiniutils-test/linux/x86_64/release/src/utils-test/test_rearrange.cc.o",
        "build/.objs/infiniutils-test/linux/x86_64/release/src/utils-test/main.cc.o",
        "build/linux/x86_64/release/libinfini-utils.a"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-Lbuild/linux/x86_64/release",
            "-s",
            "-linfini-utils",
            "-fopenmp"
        }
    }
}