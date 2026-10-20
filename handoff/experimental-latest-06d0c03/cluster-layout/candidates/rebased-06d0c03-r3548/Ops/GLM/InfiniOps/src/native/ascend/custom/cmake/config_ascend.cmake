
if(DEFINED ASCEND_HOME_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "ASCEND CANN package installation directory" FORCE)
endif()

set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME_PATH})

# Auto-detect `SOC_VERSION` from `npu-smi info` if not set externally.
# Required by `CANN`'s `ascendc.cmake` for `AscendC` kernel compilation.
if(NOT DEFINED SOC_VERSION OR "${SOC_VERSION}" STREQUAL "")
    include(${CMAKE_CURRENT_LIST_DIR}/detect_soc.cmake)
    infiniops_detect_soc(_detected_soc)
    set(SOC_VERSION "${_detected_soc}" CACHE STRING "Ascend SOC version" FORCE)
    message(STATUS "SOC_VERSION auto-set to ${SOC_VERSION}")
endif()

if(EXISTS ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_HOME_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "`ascendc_kernel_cmake` does not exist; please check whether the `CANN` package is installed.")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)


message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
