file(GLOB_RECURSE EXAMPLE_SOURCES "*.cc")

foreach(source_file ${EXAMPLE_SOURCES})
    file(RELATIVE_PATH rel_file "${CMAKE_CURRENT_SOURCE_DIR}" "${source_file}")

    get_filename_component(file_dir ${rel_file} DIRECTORY)
    get_filename_component(file_we ${rel_file} NAME_WE)

    if(file_dir)
        string(REPLACE "/" "_" target_name "${file_dir}_${file_we}")
    else()
        set(target_name ${file_we})
    endif()

    add_executable(${target_name} ${source_file})

    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME "${file_we}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${file_dir}"
    )

    target_link_libraries(${target_name} PRIVATE infiniccl)

    # Add runtime and backend dependencies for direct runtime/backend usage.
    if(WITH_NVIDIA)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    endif()

    if(WITH_ILUVATAR)
        set_source_files_properties(${source_file} PROPERTIES LANGUAGE CXX)
        set_target_properties(${target_name} PROPERTIES
            RULE_LAUNCH_COMPILE "${ILUVATAR_CUDA_COMPILER} "
        )
        target_compile_options(${target_name} PRIVATE ${ILUVATAR_CUDA_FLAGS} "-Wno-unused-command-line-argument")
        target_link_libraries(${target_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif()

    if(WITH_METAX)
        target_link_libraries(${target_name} PRIVATE ${MACA_RUNTIME_LIB})
        target_compile_options(${target_name} PRIVATE "-x" "maca")
    endif()

    if(WITH_MOORE)
        target_link_libraries(${target_name} PRIVATE ${MUSART_LIB})
        target_compile_options(${target_name} PRIVATE "-x" "musa")
    endif()

    if(WITH_CAMBRICON)
        target_link_libraries(${target_name} PRIVATE ${CAMBRICON_RUNTIME_LIB})
    endif()
    
    if(WITH_HYGON)
        target_include_directories(${target_name} PRIVATE ${HYGON_HIP_INCLUDE_DIR})
        target_compile_definitions(${target_name} PRIVATE ${HYGON_HIP_DEFINITIONS})
        target_link_libraries(${target_name} PRIVATE ${HYGON_HIP_RUNTIME_LIB})
    endif()

    if(WITH_OMPI OR WITH_MPICH)
        target_link_libraries(${target_name} PRIVATE MPI::MPI_CXX)
    endif()

    if(WITH_NCCL)
        target_link_libraries(${target_name} PRIVATE "${NCCL_LIB}")
    endif()

    if(WITH_MCCL)
        target_link_libraries(${target_name} PRIVATE "${MCCL_LIB}")
    endif()

    # Explicitly allow examples to "peek" into the internal `src` and binary dirs.
    # This is necessary because these were marked `PRIVATE` in the library's CMake.
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
        "${PROJECT_SOURCE_DIR}/src"         # For internal templates like `runtime.h`.
        "${CMAKE_BINARY_DIR}/src"           # For the generated `backend_manifest.h`.
    )
endforeach()
