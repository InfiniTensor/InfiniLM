import triton
import triton.language as tl


@triton.jit
def kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_value + other_value, mask=mask)
