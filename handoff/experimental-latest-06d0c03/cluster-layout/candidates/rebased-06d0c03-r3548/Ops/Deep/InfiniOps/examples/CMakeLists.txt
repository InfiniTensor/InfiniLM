file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS "*.cc")

# Iterate through each file and create an executable.
foreach(source_file ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${source_file} NAME_WE)

    add_executable(${example_name} ${source_file})

    target_link_libraries(${example_name} PRIVATE infiniops)

    target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    get_filename_component(example_dir ${source_file} DIRECTORY)

    target_include_directories(${example_name} PRIVATE ${example_dir})

    if(WITH_TORCH)
        foreach(_torch_dir ${TORCH_RUNTIME_DIRS})
            target_link_options(${example_name} PRIVATE "LINKER:-rpath-link,${_torch_dir}")
        endforeach()
    endif()
endforeach()
