{
    depfiles_format = "gcc",
    depfiles = "jiuge_kv_cache.o: src/models/jiuge/jiuge_kv_cache.cpp\
",
    files = {
        "src/models/jiuge/jiuge_kv_cache.cpp"
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