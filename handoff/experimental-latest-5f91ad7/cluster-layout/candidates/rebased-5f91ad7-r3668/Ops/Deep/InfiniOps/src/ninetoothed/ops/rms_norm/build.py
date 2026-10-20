import itertools

import ninetoothed
import ntops

_BLOCK_SIZES = (256, 512)

_DTYPES = ("float32", "float16", "bfloat16")

_DEFAULT_NDIMS = (2, 3, 4)

_CONFIGS = tuple(
    (
        (),
        {
            "ndim": ndim,
            "num_normalized_dims": 1,
            "input_dtype": dtype,
            "weight_dtype": dtype,
            "output_dtype": dtype,
            "block_size": block_size,
        },
        {},
    )
    for ndim, dtype, block_size in itertools.product(
        _DEFAULT_NDIMS,
        (getattr(ninetoothed, name) for name in _DTYPES),
        _BLOCK_SIZES,
    )
)


def build(output_dir):
    variant_dir = output_dir / "rms_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        ntops.kernels.rms_norm.premake,
        _CONFIGS,
        meta_parameters=("block_size",),
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_rms_norm",
        output_dir=variant_dir,
        lazy=False,
    )
