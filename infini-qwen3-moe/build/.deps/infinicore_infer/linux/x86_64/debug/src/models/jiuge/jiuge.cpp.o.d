{
    depfiles_format = "gcc",
    depfiles = "jiuge.o: src/models/jiuge/jiuge.cpp\
",
    files = {
        "src/models/jiuge/jiuge.cpp"
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