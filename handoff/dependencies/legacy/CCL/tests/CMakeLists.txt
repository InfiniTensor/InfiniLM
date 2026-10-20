add_test(
    NAME bridge_dependency_contract
    COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/bridge_dependency_contract.py"
            "${PROJECT_SOURCE_DIR}/src/CMakeLists.txt"
)

add_executable(operation_backend_selection_test
    operation_backend_selection.cc
)

target_include_directories(operation_backend_selection_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

add_test(
    NAME operation_backend_selection
    COMMAND operation_backend_selection_test
)
