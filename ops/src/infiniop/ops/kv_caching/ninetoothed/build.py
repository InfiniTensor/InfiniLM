import ninetoothed
from . import kv_caching

import infiniop.ninetoothed.build


def build():
    dtype_values = (
        ninetoothed.float16,
        ninetoothed.bfloat16,
        ninetoothed.float32,
    )

    constexpr_param_grid = {
        "emb_dim": (1, 16, 32, 64, 128, 256),
        "dtype": dtype_values,
        "block_size_m": (64,),
        "block_size_n": (64,),
    }

    infiniop.ninetoothed.build.build(
        kv_caching.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="kv_caching",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
