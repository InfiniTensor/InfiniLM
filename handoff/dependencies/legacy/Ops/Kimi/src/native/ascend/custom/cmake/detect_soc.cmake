# Auto-detect the Ascend SOC version from `npu-smi info`.
#
# `infiniops_detect_soc(<out_var>)` parses the first `910*` / `310*` entry
# in `npu-smi info` and writes `Ascend<NNNX>` into the named variable in
# the caller's scope.  Falls back to `Ascend910B4` when detection fails
# (no NPU on the host, `npu-smi` missing, output format mismatch).
#
# Called from both `src/CMakeLists.txt` (outer `pip install` build, to
# forward `SOC_VERSION` to the standalone `build.sh` invocation) and
# `src/native/ascend/custom/cmake/config_ascend.cmake` (the sub-build driven
# by that `build.sh`).

function(infiniops_detect_soc out_var)
    execute_process(
        COMMAND bash -c "npu-smi info 2>/dev/null | awk '/910B|910A|310/ {for (i=1;i<=NF;i++) if ($i ~ /^(910|310)/) {print \"Ascend\" $i; exit}}'"
        OUTPUT_VARIABLE _detected
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_detected)
        set(${out_var} "${_detected}" PARENT_SCOPE)
    else()
        set(${out_var} "Ascend910B4" PARENT_SCOPE)
    endif()
endfunction()
