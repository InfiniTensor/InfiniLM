target("infiniutils-test")
    set_kind("binary")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_warnings("all", "error")
    set_languages("cxx17")
    
    add_files(os.projectdir().."/src/utils-test/*.cc")

target_end()
