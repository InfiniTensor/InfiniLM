# Contributing Guide

For development setup, see [Development Guide](#development-guide) below.

## Code

Please review these details before committing, especially for AI-generated code.

### General

1. Keep changes minimal — do not add what is not necessary.
2. Comments are not always better when abundant. Ideally, the code should be self-explanatory.
3. Files must end with a newline.
4. Use Markdown syntax (backtick-fenced) for identifiers in comments and error messages.
5. Comments and error messages must be in English.
6. Comments and error messages should follow the language's conventions first. If the language does not specify, use complete sentences — capitalize the first letter and end with punctuation.

### C++

1. Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly. Use the default `.clang-format`.
2. When defining function parameters, consistency with NCCL has the highest priority. 
    - For functions that correspond to NCCL, parameter order must match the NCCL interface exactly. 
    - For all other functions, follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html#Inputs_and_Outputs) for parameter ordering (inputs before outputs), unless doing so would reduce clarity, cause significant inconvenience, or break consistency with related interfaces. 
3. Do not use exceptions. Use `assert` for error handling — debug builds will trigger assertion messages (which should include `__FILE__`, `__LINE__`, `__func__` at minimum), and release builds will have assertions compiled out.
4. Error and warning messages follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages).
5. Operation file naming (excluding extension): especially for the ones under `src/base/`, file names must follow NCCL naming semantics. 
    - If an operation corresponds to an NCCL API, the file name must be the snake_case decomposition of the NCCL function name, splitting words at CamelCase boundaries.
    - The name must contain exactly the same words as the NCCL function, with no additions or omissions.
    - Example: for all-reduce, NCCL provides: `ncclAllReduce`, then the corresponding file name should be `all_reduce`.
6. Initializer list order must match member declaration order.
7. One blank line between classes, between classes and functions, and between functions.
8. One blank line between each member (including both functions and variables) within a class.
9. One blank line before and after the contents of a namespace.

### Python

Follow [PEP 8](https://peps.python.org/pep-0008/) as the primary style guide. For anything PEP 8 does not cover in detail, refer to the [GDScript style guide](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)—while it targets a different language, its non-syntax conventions are still applicable.

#### Additional Rules

1. **Comments** should be complete English sentences, starting with a capital letter and ending with punctuation. Use Markdown syntax when referencing code within comments.

2. **Error messages and framework conventions:** When a framework has an established convention (e.g., `pytest.skip` messages are typically lowercase without a trailing period), follow that convention. Otherwise, use the same rules as comments.

3. **Function signatures:** If a function has no docstring or comment, do not add a blank line between the function signature and the function body.

4. **Blank lines around control flow:** Add a blank line before and after `if`, `for`, and similar statements.

5. **Return statements:** Add a blank line before a `return` statement, unless it directly follows a control flow statement like `if` or `for`.

6. **Docstrings:** Follow [PEP 257](https://peps.python.org/pep-0257/) conventions.

## Commits

Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/).

## Pull Requests

1. Small PRs should be squashed. Large PRs may keep multiple commits, but each commit must be meaningful and well-formed.
2. PR titles follow the same Conventional Commits format as commit messages.
3. Before merging (or after each major change), ensure that all example programs build and run successfully on at least one supported heterogeneous cluster setup, and include the test environment details and results in the pull request. Whenever possible, validate on a cluster covering all supported platforms and backends. 

## Branches

Branch names use the format `<type>/xxx-yyyy-zzzz`, where `<type>` matches the PR title's Conventional Commits type, and words are joined with hyphens.

# Development Guide

## Some Code Explanations

### `TypeMap` and `DataTypeMap`

Since `DataType` is an enum used to represent data types generically, we often need to map between `DataType` and native C++ types (e.g. `float`, `int32_t`).

- **`TypeMap`**: maps `DataType` to native types. Use the alias `TypeMapType` to get the type directly, e.g. `TypeMapType<dev, DataType::kFloat32>` is `float`. Note, the first template argument is a `Device::Type` since data types like float16 and bfloat16 are not the same across the platforms. Thus, a `Device::Type` is required to specify which native type a `DataType` maps to. 
- **`DataTypeMap`**: maps native types back to `DataType`. Use the alias `DataTypeMapValue`, e.g. `DataTypeMapValue<float>` is `DataType::kFloat32`.

### `DispatchFunc`

`DispatchFunc` is the runtime dispatch interface defined in `dispatcher.h`. It supports arbitrary types, multi-dispatch, and mixed-type dispatch with any return type.

#### Basic Usage

```cpp
DispatchFunc</* supported types */>(
    /* runtime value to dispatch on */,
    /* lambda with dispatched logic */,
    /* context string for error messages (recommended) */,
    /* forwarded args for the lambda (optional) */
);
```

#### Single-Type Dispatch (`Device::Type`)

```cpp
DispatchFunc<Device::Type::kCpu, Device::Type::kNvidia>(
    Device::Type::kNvidia,
    [](auto tag) {
      constexpr Device::Type Dev = decltype(tag)::value;
    },
    "DeviceTest");
```

#### Single-Type Dispatch (`DataType`)

```cpp
DataType dtype = DataType::kFloat32;
DispatchFunc<Device::Type::Cpu, FloatTypes>(
    dtype,
    [](auto tag) {
      using T = typename decltype(tag)::type;
      // Use T as the resolved native type.
    },
    "DataType Dispatch");
```

Dispatching `DataType` is a little bit special. 

1. Due to the previously mentioned `TypeMap` reason, a `Device::Type` is needed as the first template argument;

2. Since `DataType` is frequently used, the supported type list can use predefined shorthands from `data_type.h` (e.g. `FloatTypes` = `List<DataType::kFloat32, DataType::kFloat64>`). To combine shorthands, use `ConcatType` from `common/traits.h`:

```cpp
DispatchFunc<ConcatType<List<DataType::kFloat16>, FloatTypes>>(...);
```

#### Single-Type Dispatch (Custom Types)

For types other than `DataType` and `Device::Type`, pass the type as the first template argument:

```cpp
DispatchFunc<QuantMode, QuantMode::kNone, QuantMode::kWeightOnly>(
    QuantMode::kWeightOnly,
    [](auto tag) {
      constexpr QuantMode M = decltype(tag)::value;
    },
    "QuantDispatch");
```

This also works for native types like `int` (e.g. block sizes):

```cpp
DispatchFunc<int, 128, 256, 512, 1024>(
    runtime_block_size,
    [](auto tag) {
      constexpr int BlockSize = decltype(tag)::value;
    },
    "BlockSizeDispatch");
```

#### Multi-Dispatch (Same Type)

Use `List` boundaries to separate supported sets for each dispatched value. Pass runtime values in an initializer list:

```cpp
DispatchFunc<List<Device::Type::kCpu, Device::Type::kNvidia>,
             List<Device::Type::kAscend, Device::Type::kMetax>>(
      {Device::Type::kNvidia, Device::Type::kMetax},
      [](auto tag1, auto tag2) {
        constexpr Device::Type D1 = decltype(tag1)::value;
        constexpr Device::Type D2 = decltype(tag2)::value;
      },
      "MultiDeviceTest");
```

Similarly, `DataType` requires a `Device::Type` at the front: 

```cpp
DispatchFunc<Device::Type::kCpu, FloatTypes, List<DataType::kInt32, DataType::kInt64>>(
    {DataType::kFloat64, DataType::kInt32},
    [](auto tag1, auto tag2) {
      using T1 = typename decltype(tag1)::type;
      using T2 = typename decltype(tag2)::type;
    },
    "MultiDataTypeTest");
```

#### Mixed Multi-Type Dispatch

When dispatching different types simultaneously (e.g. `DataType` + `Device::Type`, or `DataType` + block size), cast values to `int64_t` and use `ListGet<N>` from `common/traits.h`:

```cpp
DispatchFunc<FloatTypes, List<Device::Type::kCpu, Device::Type::kNvidia>>(
    {static_cast<int64_t>(DataType::kFloat32),
     static_cast<int64_t>(Device::Type::kNvidia)},
    [](auto list_tag) {
      constexpr DataType DT = static_cast<DataType>(ListGet<0>(list_tag));
      constexpr Device::Type Dev = static_cast<Device::Type>(ListGet<1>(list_tag));
      using T = TypeMapType<Device::Type::kCpu, DT>;
    },
    "MixedDispatch");
```

Note that in mixed multi-type dispatch, `DataType` is not treated specially. Therefore, we neither should nor can place `Device::Type` at the front of the `DataType` list. Inside the lambda, we obtain it as a `DataType` and then convert it to the native type if needed. 

If `DT` is not used within the lambda, you can inline its definition directly into the `using T = ...` statement, like this: 

```cpp
using T = TypeMapType<Device::Type::kCpu, ListGet<0>(list_tag)>;
```

## Troubleshooting
### 1. `icclrun` fails on the first node with a `VAR:type=value` CMake error
```bash
[*] Orchestrating nvidia on localhost...
CMake Error: Parse error in command line argument: ...
 Should be: VAR:type=value
CMake Error: Run 'cmake --help' for all supported options.
Traceback (most recent call last):
...
```
If you encounter this error, make sure you are using the correct `cluster.yaml` and that all required fields are filled in using the proper format.
