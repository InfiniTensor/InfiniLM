# Find the Python binary.
find_program(PYTHON_EXECUTABLE NAMES python3)

if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message(FATAL_ERROR "`python3` is not found; install Python first.")
endif ()

# Get `torch`, `torch_npu`, and `pybind11` paths via a Python helper.
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c"
        "import torch; import torch_npu; import os; import pybind11; import sysconfig;
torch_dir = os.path.realpath(os.path.dirname(torch.__file__));
torch_npu_dir = os.path.realpath(os.path.dirname(torch_npu.__file__));
pybind11_dir = os.path.realpath(os.path.dirname(pybind11.__file__));
abi_enabled=torch.compiled_with_cxx11_abi();
python_include_dir = sysconfig.get_path('include');
print(torch_dir, torch_npu_dir, pybind11_dir, abi_enabled, python_include_dir, end='');
quit(0)
        "
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_VARIABLE OUTPUT_ENV_DEFINES)

# Abort if the Python helper failed.
if (NOT ${EXEC_RESULT} EQUAL 0)
    message(FATAL_ERROR "Failed to run Python script to probe env vars like `TORCH_DIR`.")
else ()
    message(STATUS "Python probe succeeded; output string is [${OUTPUT_ENV_DEFINES}].")
endif ()

# Extract `TORCH_DIR`.
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $1}'"
        OUTPUT_VARIABLE TORCH_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract `TORCH_NPU_DIR`.
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $2}'"
        OUTPUT_VARIABLE TORCH_NPU_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract `PYBIND11_DIR`.
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $3}'"
        OUTPUT_VARIABLE PYBIND11_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract the PyTorch C++11 ABI flag.
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $4}'"
        OUTPUT_VARIABLE TORCH_API_ENABLED
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Extract `PYTHON_INCLUDE_DIR`.
execute_process(
        COMMAND sh -c "echo \"${OUTPUT_ENV_DEFINES}\" | awk '{print $5}'"
        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
        RESULT_VARIABLE EXEC_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "SOC_VERSION=${SOC_VERSION}")
message(STATUS "TORCH_DIR=${TORCH_DIR}")
message(STATUS "TORCH_NPU_DIR=${TORCH_NPU_DIR}")
message(STATUS "PYBIND11_DIR=${PYBIND11_DIR}")
message(STATUS "PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}")

# Set `_GLIBCXX_USE_CXX11_ABI` to match the PyTorch build.
if (${TORCH_API_ENABLED} STREQUAL "True")
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=1")
else ()
    add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
    message(STATUS "_GLIBCXX_USE_CXX11_ABI=0")
endif ()
