from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _unwrap(value):
    return value._underlying


def add(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.add(_unwrap(input), _unwrap(other)))
    _infinicore.add_(_unwrap(out), _unwrap(input), _unwrap(other))
    return out


def cat(tensors, dim=0, *, out=None):
    underlying = [_unwrap(tensor) for tensor in tensors]
    if out is None:
        return Tensor(_infinicore.cat(underlying, dim))
    _infinicore.cat_(_unwrap(out), underlying, dim)
    return out


def matmul(input, other, *, alpha=1.0, out=None):
    if out is None:
        return Tensor(_infinicore.matmul(_unwrap(input), _unwrap(other), alpha))
    _infinicore.matmul_(_unwrap(out), _unwrap(input), _unwrap(other), alpha)
    return out


def linear(input, weight, bias=None, *, alpha=1.0, out=None):
    raw_bias = None if bias is None else _unwrap(bias)
    if out is None:
        return Tensor(
            _infinicore.linear(_unwrap(input), _unwrap(weight), raw_bias, alpha)
        )
    _infinicore.linear_(_unwrap(out), _unwrap(input), _unwrap(weight), raw_bias, alpha)
    return out


def embedding(input, weight, *, out=None):
    if out is None:
        return Tensor(_infinicore.embedding(_unwrap(input), _unwrap(weight)))
    _infinicore.embedding_(_unwrap(out), _unwrap(input), _unwrap(weight))
    return out


def rms_norm(input, weight, epsilon=1e-5, *, out=None):
    if out is None:
        return Tensor(_infinicore.rms_norm(_unwrap(input), _unwrap(weight), epsilon))
    _infinicore.rms_norm_(_unwrap(out), _unwrap(input), _unwrap(weight), epsilon)
    return out


def causal_softmax(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.causal_softmax(_unwrap(input)))
    _infinicore.causal_softmax_(_unwrap(out), _unwrap(input))
    return out


def silu(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.silu(_unwrap(input)))
    _infinicore.silu_(_unwrap(out), _unwrap(input))
    return out


def silu_and_mul(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.silu_and_mul(_unwrap(input)))
    _infinicore.silu_and_mul_(_unwrap(out), _unwrap(input))
    return out


def random_sample(logits, random_val, topp, topk, temperature, *, out=None):
    arguments = (_unwrap(logits), random_val, topp, topk, temperature)
    if out is None:
        return Tensor(_infinicore.random_sample(*arguments))
    _infinicore.random_sample_(_unwrap(out), *arguments)
    return out
