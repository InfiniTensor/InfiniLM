# Operator API Alignment

This guide defines how InfiniOps operator APIs should be compared with and
aligned to established open-source frameworks. It applies when adding a new
operator and when auditing an existing operator under `src/base/`.

## Select the Alignment Target

Choose one named upstream interface before designing or reviewing the
InfiniOps API. Do not combine convenient pieces from unrelated layers into a
new signature.

Use the following order:

1. Use the PyTorch public Python API when the same operation exists there.
2. For serving-specific operations not exposed by PyTorch, use a public vLLM
   Python wrapper, then a public SGLang Python wrapper.
3. For general graph operations with no suitable framework API, use the ONNX
   operator schema. `Gemm` is an example of a valid ONNX-aligned operator.
4. For library-specific operations, use the library's public wrapper, such as
   FlashAttention, before using its private C++ or CUDA entry point.
5. Use a CUDA or vendor API only when no higher-level stable interface exists.
6. If no stable public target exists, classify the operator as custom instead
   of claiming alignment.

The selected layer matters. A C++ registration schema can confirm types and
dispatch behavior, but it does not replace an existing public Python API. This
rule applies to PyTorch, vLLM, and SGLang alike.

For PyTorch:

- compare against `torch.*`, `torch.nn.functional.*`, or the documented
  Tensor method first
- use the ATen schema to confirm types or implement an ATen backend
- do not use an ATen-only argument order to override the public Python order

Generated or explicitly internal PyTorch-derived operators may target an ATen
schema directly. Label them as ATen-schema aligned, not Python-API aligned.

For example, the public
[`torch.nn.functional.embedding`](https://github.com/pytorch/pytorch/blob/a57db29aa6d98118fdd27a7d509ab3525a750669/torch/nn/functional.py#L2509)
interface starts with `input, weight`. An ATen schema that names or orders
those tensors differently is not sufficient evidence for reversing these two
leading public parameters.

When upstream frameworks expose different contracts, choose the framework that
owns the operation or is already used by its consumers. State that choice in
the pull request instead of describing the operator as aligned with every
framework.

## What Counts as Aligned

An operator is interface-aligned when at least one canonical InfiniOps overload
has all of the following properties:

- Its name identifies the same operation as the selected target.
- Every material upstream input and attribute is representable.
- Parameters preserve upstream order within the InfiniOps input, attribute,
  and output groups.
- C++ types preserve distinctions between tensors, scalars, optional values,
  sequences, and framework objects.
- Every value in the upstream maximal return contract is representable.
- Every difference is an accepted C++ adaptation below and is documented.

Use these review classifications:

| Classification | Meaning |
| --- | --- |
| Aligned | The canonical overload matches the selected target, with no differences. |
| Aligned with C++ adaptation | Only accepted InfiniOps adaptations are present. |
| Convenience gap | The canonical contract is complete, but a default, omission, scalar, or reduced-return path is absent. |
| Needs remediation | The name differs, a material parameter is absent or incorrectly ordered, its type changes the contract, or a maximal result is missing. |
| Custom | No stable public upstream target was found. |

A finding may additionally be marked as deferred to record scheduling. Deferred
is not an alignment classification: retain the underlying `Convenience gap` or
`Needs remediation` status and record why work was postponed.

A parameter rename alone is normally not a material C++ interface difference,
but new code should still use the selected upstream name. A rename becomes
material when it obscures a different role or semantic meaning.

## Accepted InfiniOps C++ Adaptations

InfiniOps is a C++ operator library with caller-provided output tensors. Exact
Python syntax is therefore not always the correct C++ shape.

### Parameter grouping

Follow `CONTRIBUTING.md`:

1. tensor and framework-object inputs
2. attributes
3. output tensors

Preserve upstream relative order within each group. Optional tensor inputs are
still inputs and must not be moved behind scalar attributes. Mutable outputs
must be at the end even when a lower-level registration places them first.

For example, a vLLM operation with output buffers before its logical input can
be represented with the same inputs and attributes followed by the outputs.
Identify this as the repository ordering rule, not an exact positional match.

### Explicit output tensors

An upstream returned tensor may be represented as a trailing writable `Tensor`.
A returned tuple may be represented as multiple trailing output tensors in the
same order.

This does not permit dropping a return value or changing an always-present
result to an unrelated optional input.

### Explicit attributes

C++ callers may be required to pass an attribute that has a Python default. The
canonical interface remains aligned when the parameter exists with the same
role and type. Missing omission paths are convenience gaps and can later be
added with a delegating overload or `std::optional`.

Keyword-only markers and Python `None` defaults do not need literal C++ syntax.
Use `std::optional` where optionality is part of the canonical contract.

### C++ type mapping

Use the narrowest existing InfiniOps or standard C++ type that preserves the
upstream distinction:

| Upstream role | Preferred C++ representation |
| --- | --- |
| Tensor | `Tensor` |
| Optional tensor | `std::optional<Tensor>` |
| Boolean, integer, or floating attribute | `bool`, `int64_t`, or `double` as appropriate |
| Shape or integer sequence | `std::vector<int64_t>` or an existing shape type |
| Optional scalar or sequence | `std::optional<T>` |
| Data type | `DataType` or `std::optional<DataType>` |
| Writable result | trailing `Tensor` or `std::optional<Tensor>` |

Do not erase a Tensor/scalar distinction merely to reduce overload count. Do
not introduce a new framework-object class, such as a generator or storage
wrapper, until its ownership and cross-backend contract are understood.

Writable optional outputs use `std::optional<Tensor>`, not
`std::optional<const Tensor>`.

## Names and Custom Operators

Operator names are part of the public interface and matter more than C++
parameter spelling. Use the selected upstream operation name, converted only
to repository file and class naming conventions.

Material naming questions include `fused_add_rms_norm` versus
`add_rms_norm`, and a standard `relu` versus an InfiniLM-specific ReLU
contract. Do not rename an operation solely because a private kernel or ATen
symbol uses a different name.

An operation that exists only for InfiniLM compatibility and has no stable
target should retain an `Infinilm` suffix until a canonical replacement
exists. Do not move such operations into `internal` merely to remove the
suffix. `internal` naming and namespaces are reserved for PyTorch-derived
internal operators.

Migrate these compatibility operators gradually: add a canonical operator
aligned with an open-source target, move callers to it, explicitly deprecate
the `Infinilm` API, and remove it only in a later breaking change.

## Overload Design

Prefer one canonical overload. Add another only when it represents:

- a distinct public upstream contract, such as Tensor and scalar forms
- a necessary optional framework object that the canonical overload cannot
  express
- a compatibility path for an existing InfiniOps interface
- a small delegating convenience path with demonstrated value

Do not add every permutation of defaults, Tensor/scalar combinations, or
argument order. Upstream internal ATen overloads do not all need InfiniOps
counterparts.

When a new canonical overload replaces an existing interface:

1. keep the previous overload temporarily
2. mark it with `[[deprecated("...")]]` when supported
3. state the replacement and intended removal in the diagnostic
4. retain coverage for both canonical and deprecated paths
5. remove the old path later in a separately reviewed breaking change

A source comment alone is not a sufficient deprecation signal.

If a derived implementation overrides only one overload, use
`using Foo::operator();` where needed so inherited overloads remain visible.

Adding an operator to `scripts/torch_ops.yaml` is not an alignment requirement.
Do so only when the generated ATen backend is needed and the selected schema
maps correctly. See [Adding ATen-backed operators](aten-operators.md).

## Return Contracts

Review results independently from input parameters.

### Fixed multiple results

If the target always returns multiple values, the canonical InfiniOps interface
must expose every value as a trailing output. A `*_with_indices` pooling
operation with a fixed `(output, indices)` result must accept both outputs.

### Conditional auxiliary results

If an attribute enables auxiliary results, the overload containing that
attribute must provide corresponding optional output buffers. Keep a simpler
out-only overload when it represents the normal path; do not add optional
auxiliary outputs to every basic overload.

Optional outputs should be present exactly when the controlling return flag
requests them. This applies to interfaces such as FlashAttention's LSE or
attention-probability results.

### Reduced-return convenience paths

When an InfiniOps interface already exposes the maximal result set, the absence
of a convenience path returning fewer values is a convenience gap, not a
missing maximal return contract. Track it separately and do not expand the API
unless callers need it.

When applying these rules, record the selected upstream interface and any
intentional C++ adaptations in the pull request. Follow `CONTRIBUTING.md` for
compatibility, testing, and validation requirements.
