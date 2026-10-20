# InfiniCCL

InfiniCCL is a unified, cross-platform collective communication library designed to provide a unified interface for distributed deep learning and high-performance computing workloads across heterogeneous hardware architectures.

## Prerequisites

### Software Requirements
- C++17-compatible compiler

- CMake ≥ 3.18

- Python ≥ 3.10 

    - Earlier versions MAY work but are not guaranteed

- Involved Hardware-specific SDKs (e.g. CUDA Toolkit)

- Involved Backend Libraries (e.g., OpenMPI, NCCL)

### System Configuration
The participating nodes must meet the following requirements:
- Network connectivity between all nodes

- Passwordless SSH access

- Consistent library and binary paths across the cluster (e.g., shared filesystem such as NFS). 
    - Note: Using inconsistent file paths across nodes has not yet been fully validated and may cause runtime failures during orchestration. 

## Key Features

* **Platform Agnostic**: NCCL‑like, unified C API for all supported platforms.
* **Heterogeneous Communication**: Supports communication operations across different platforms and accelerators.
* **Platform Auto-Detection**: Automatically detects the underlying hardware without manual specification.
* **Smart Orchestration**: Includes `icclrun`, a lightweight launcher for automated multi-node building, installation, and execution.

## Building and Installation

InfiniCCL can be built either manually with CMake or using the provided `build.sh` script for a single‑command, environment‑aware installation. Currently InfiniCCL only offers install-from-source option.

To start, download InfiniCCL using `git`: 

```bash
# Clone the repository
git clone https://github.com/InfiniTensor/InfiniCCL.git
cd InfiniCCL
```

There are two methods to build InfiniCCL. 

### Method 1 - Quick Build with `build.sh` (Recommended)

```bash
# Grant execution permission to the script
chmod +x ./scripts/build.sh

# Build and install to default location (~/.infini)
./scripts/build.sh
```
To specify a custom install path (e.g., NFS shared directory)

```bash
INSTALL_PREFIX=/nfs/shared/infini ./scripts/build.sh
```

To set any CMake flag or option, simply add it after the script (See the end of this section for all the available options)

```bash
# For instance, explicitly turn on `NCCL`
./scripts/build.sh -DWITH_NCCL=ON
```

After the script completes, you will see output similar to:

```
========================================================
 Starting InfiniCCL Build
 Build Directory:  build
 Install Prefix:   /nfs/shared/infini
========================================================
-- No backend specified. Defaulting to WITH_OMPI=ON
-- Auto-detecting available devices...
-- Auto-detected NVIDIA environment.
-- InfiniCCL Config: Devices [cpu, nvidia] | Backends [ompi]
...
-- Installing: /nfs/shared/infini/lib/libinfiniccl.so
...
========================================================
 Build and Install Finished!
 Binaries:  /nfs/shared/infini/bin
 Libraries: /nfs/shared/infini/lib
 Headers:   /nfs/shared/infini/include
========================================================
 NOTE: If this is your first install, run: source ~/.bashrc
 You can now run 'icclrun' from anywhere.
```

Note: Running `source ~/.bashrc` (or restarting your terminal) is required only once to add `icclrun` to your `PATH`.

---

### Method 2 - Manual Build with CMake

InfiniCCL can be built and installed manually.

```bash
# Build and Install InfiniCCL
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install [options]
cmake --build . -j $(nproc)
cmake --install .

# Configure the Enviroment Variables
# It is recommended to add these to your `~/.bashrc` for persistence.
export INFINICCL_ROOT="/path/to/InfiniCCL"
export PATH="/path/to/install/bin:$PATH"
export LD_LIBRARY_PATH="/path/to/install/lib:$LD_LIBRARY_PATH"
```

Example Configurations: `NVIDIA` + `OpenMPI`

```bash
cmake .. -DWITH_NVIDIA=ON -DWITH_OMPI=ON
```

---

<details>
<summary><b>📋 Full list of CMake options (Click to expand)</b></summary>

### InfiniCCL‑Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| **Device (Hardware) Options** |||
| `WITH_NVIDIA`     | Enable NVIDIA GPU support | `OFF` |
| `WITH_ILUVATAR`   | Enable Iluvatar GPU support | `OFF` |
| `WITH_METAX`      | Enable MetaX GPU support  | `OFF` |
| `WITH_MOORE`      | Enable Moore Threads GPU support  | `OFF` |
| `WITH_CAMBRICON`  | Enable Cambricon MLU support  | `OFF` |
| `WITH_HYGON`      | Enable HYGON DCU support  | `OFF` |
| `WITH_CPU`        | CPU support (always enabled) | `ON` (internal, not user‑settable) |
| **Backend (Communication) Options** |||
| `WITH_OMPI`   | Enable OpenMPI backend | `ON` if no backend specified, otherwise `OFF` |
| `WITH_MPICH`  | Enable MPICH backend | `OFF` |
| `WITH_NCCL`   | Enable NCCL backend | `OFF` |
| `WITH_MCCL`   | Enable MCCL backend | `OFF` |
| **Miscellaneous** |||
| `AUTO_DETECT_DEVICES` | Automatically detect available devices and enable corresponding support | `ON` |
| `AUTO_DETECT_BACKENDS` | Automatically detect available communication backends and enable corresponding support | `OFF` |
| `BUILD_EXAMPLES` | Build internal example programs | `ON` |

### Standard CMake Options (not InfiniCCL‑specific)

These options are available in any CMake project and can be passed during configuration:

| Option | Description | Typical Default |
|--------|-------------|----------------|
| `CMAKE_INSTALL_PREFIX` | Installation directory | `/usr/local` (set via `-DCMAKE_INSTALL_PREFIX=...` or environment) |
| `CMAKE_BUILD_TYPE` | Build type (`Release`, `Debug`, `RelWithDebInfo`, `MinSizeRel`) | (empty, but `build.sh` sets `Release`) |

> **Notes**:
> - `AUTO_DETECT_DEVICES` overrides device options like `WITH_NVIDIA` when detection succeeds.
> - `AUTO_DETECT_BACKENDS` overrides backend options like `WITH_OMPI` when detection succeeds.

</details>

## Using the Library

To compile and execute any program that uses InfiniCCL, we can simply use `icclrun` from a single node to launch, compile, and execute the program across a cluster. 

To achieve this, a `cluster.yaml` is required to be filled. This is the configuration file for `icclrun`. [`examples/cluster.yaml`](examples/cluster.yaml) is an example template. 

After having a successful build and a complete `cluster.yaml`, we are ready for compiling and executing a distributed program across the cluster. 

<details>
<summary><b>📋 Full list of `cluster.yaml` fields (Click to expand)</b></summary>

| Field | Required | Scope | Available Values / Meaning |
|-------|----------|-------|----------------------------|
| `common_dir` | Yes | Global | Root directory of the project source tree. For internal examples, set this to the InfiniCCL source root. For external programs, set this to the external project root. |
| `common_user` | No | Global | Default SSH user for remote nodes. If omitted and no node-level `user` is set, `root` is used. |
| `install_dir` | No | Global | Base directory for InfiniCCL installation artifacts. If omitted, `common_dir` is used. Libraries are installed under `<install_dir>/install/<type>`. |
| `cmake_flags` | No | Global | CMake options applied to every node during `--build`, such as `-DAUTO_DETECT_BACKENDS=OFF`. Node-level `cmake_flags` override this value for that node. |
| `backend_args` | No | Global | Extra arguments passed to the selected backend launcher. Use a YAML mapping from a flag to a scalar or list of values, for example `--mca: ["pml ucx", "btl ^openib"]` for OpenMPI. |
| `backend_env` | No | Global | Environment variables exported for launcher/runtime setup on every node. Node-level `backend_env` overrides these values; path-like variables (`LD_LIBRARY_PATH`, `PATH`, `CPATH`, `LIBRARY_PATH`) are prepended to the global value. |
| `launcher_script` | No | Global | Path to a custom wrapper script. If omitted, `icclrun` generates `build/run_wrapper.sh` from `cluster.yaml`. For MPI launchers, the script path must be executable from every participating node. |
| `nodes` | Yes | Global | List of participating nodes. Each entry describes one host and its build/runtime settings. |
| `nodes[].ip` | Yes | Node | Node IP address or hostname. Use `localhost` or `127.0.0.1` for the local node. |
| `nodes[].user` | No | Node | SSH user for this node. Overrides `common_user`. |
| `nodes[].dir` | No | Node | Node-specific project source directory. Overrides `common_dir` for this node and is useful when the project path differs across hosts. |
| `nodes[].type` | Yes | Node | Architecture/build label used in build, install, and wrapper paths. Common values include `cpu`, `nvidia`, `iluvatar`, `metax`, `moore`, `cambricon`, and `hygon`. |
| `nodes[].slots` | No | Node | Number of processes to launch on this node. This usually matches the number of devices assigned to the node. Defaults to `8`. |
| `nodes[].cmake_flags` | No | Node | Node-specific CMake options used during `--build`, such as `-DUSE_CUDA=ON` or `-DUSE_MACA=ON`. Overrides global `cmake_flags` for this node. |
| `nodes[].backend_env` | No | Node | Node-specific runtime environment variables, such as `CUDA_VISIBLE_DEVICES`, `UCX_TLS`, or `UCX_NET_DEVICES`. Overrides or prepends to global `backend_env` values for this node. |

> **Notes**:
> - `backend_args` and launcher-level environment forwarding are used by `--launcher ompi` and `--launcher mpich`.
> - `--launcher none` still uses the generated or configured wrapper script, but does not invoke `mpirun`.
> - Unknown YAML keys are ignored by the current launcher implementation.

</details>


### 1. Run Internal Examples

#### 1.1. Run a Single Example
The repository organizes example suites by collective paradigm layout under `examples/`.

To run an internal example program (e.g., `examples/mpi/all_reduce.cc`), just run: 

```bash
# Build and run the executable across the cluster based on the config specified in your `cluster.yaml`
icclrun --config [path-to-your-cluster.yaml] --build mpi/all_reduce [program args]
```

- `--config / -c` : Path to the cluster YAML file.
- `--build` : Instructs `icclrun` to compile the library on each node before execution. If omitted, `icclrun` assumes the library is already installed at a consistent location.
- `--launcher` : Specify the backend launcher to be used. Default to `ompi`.

The executable (e.g., `mpi/all_reduce`) and its arguments follow the options.

For more details, run:

```bash
icclrun --help
```

If everything is correctly set up, the example program should have just been launched and executed across the cluster you specified in `cluster.yaml`!

#### 1.2. Automated Multi-Collective Verification Harness
To run and log multiple example programs consecutively, use the provided automated batch verification script `scripts/run_examples.py`. This tool runs multiple validation profiles, monitors exit codes, captures chronological trace logs, and optimizes away redundant compilation passes. 

For the first time using the script, grant it execution permission:

```bash
# Grant execution permission to the script
chmod +x ./scripts/run_examples.sh
```

For all the options of the script, see:

```bash
./scripts/run_examples.py --help
```

### 2. Run a Custom User Program

Within the program, just include:

```cpp
#include <infiniccl/infiniccl.h>
```

#### Recommended: `find_package(InfiniCCL)`

The simplest way to consume InfiniCCL is through `find_package`. This pulls in the
imported target `InfiniCCL::infiniccl`, and there is no need to hand-wire include
paths, the `.so`, or the RPATH yourself.

```cmake
cmake_minimum_required(VERSION 3.18)
project(UserApp LANGUAGES CXX)

find_package(InfiniCCL REQUIRED)

add_executable(app main.cc)
target_link_libraries(app PRIVATE InfiniCCL::infiniccl)
```

Point CMake at the InfiniCCL install prefix with any of the usual mechanisms:

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/install   # e.g. ~/.infini
# or:  -DInfiniCCL_ROOT=/path/to/install
```

There is no need to resolve hardware/backend dependencies (e.g., CUDA, MPI, NCCL, …)
just to use InfiniCCL. However, if your own program calls those APIs directly
(e.g. allocating GPU buffers or calling MPI), you must link them in addition to
`InfiniCCL::infiniccl`.

#### Alternative: manual linking

If you prefer not to use the config package, set the environment variable
`INFINICCL_INSTALL` to the InfiniCCL installation directory (e.g. `~/.infini` or
a shared NFS path) and link directly to the library and headers:

```cmake
cmake_minimum_required(VERSION 3.18)
project(UserApp LANGUAGES CXX)

# Path to InfiniCCL installation (set via environment)
set(INFINI_INSTALL_DIR $ENV{INFINICCL_INSTALL})

# Your application
add_executable(app main.cc)

# Direct linking to InfiniCCL
target_include_directories(app PRIVATE "${INFINI_INSTALL_DIR}/include")
target_link_libraries(app 
    PRIVATE 
    "${INFINI_INSTALL_DIR}/lib/libinfiniccl.so" 
    # other necessary libraries
    # ...
)

# RPATH to find `libinfiniccl.so` at runtime
set_target_properties(app PROPERTIES 
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${INFINI_INSTALL_DIR}/lib"
)
```

Then build and run with `icclrun` similar to internal examples:

```bash
icclrun --config [path-to-your-cluster.yaml] --build app [program args]
```

If You are not using CMake, you can compile and link the program manually similar to other dependencies, for instance with these options: 

```bash
-I${INFINI_INSTALL}/include \
-L${INFINI_INSTALL}/lib \
-Wl,-rpath,${INFINI_INSTALL}/lib \
-linfiniccl \
```

If without `-Wl, -rpath`:
```bash
export LD_LIBRARY_PATH=${INFINI_INSTALL}/lib:$LD_LIBRARY_PATH
```

## Supported Platforms and Backends

<details>
<summary><b>📋 Supported runtime device platforms (Click to expand)</b></summary>

| Platform | Support Level | Notes |
|----------|---------------|-------|
| **CPU**        | Partial | Runtime available, suppot OpenMPI backend, but no pure CPU collective operations yet. Planned for future releases. |
| **NVIDIA**     | Full | Requires CUDA Toolkit. |
| **Iluvatar**   | Full | Requires Iluvatar CoreX SDK. |
| **MetaX**      | Full | Requires MACA SDK and `MACA_PATH` (default `/opt/maca`) to be set. |
| **Moore Threads** | Full | Requires MUSA SDK and at least one of `MACA_ROOT` (default `/usr/local/musa`), `MACA_PATH`, and `MUSA_HOME` to be set. |
| **Cambricon**  | Full | Requires CNToolKit and `NEUWARE_HOME` to be set. |
| **HYGON**      | Full | Requires HYGON DTK and HYHAL. |

</details>

<details>
<summary><b>📋 Supported communication backends (Click to expand)</b></summary>

| Backend | Support Level | Required CMake Option | Dependencies |
|---------|---------------|----------------------|---------------|
| **OpenMPI** | Full | `WITH_OMPI=ON` | The default backend. Requires the OpenMPI development package.|
| **MPICH** | Full | `WITH_MPICH=ON` | Requires the MPICH development package.|
| **NCCL** | Partial | `WITH_NCCL=ON` | Requires NVIDIA or Iluvatar NCCL. Currently available when `WITH_NVIDIA=ON` or `WITH_ILUVATAR=ON`.|
| **MCCL** | Partial | `WITH_MCCL=ON` | Requires MetaX or Moore MCCL. Currently available when `WITH_METAX=ON` or `WITH_MOORE=ON`.|

</details>

## License
InfiniCCL is distributed under the [LICENSE](LICENSE) file in the root of the repository.

## Contributing
Contributions are welcome! Please refer to the contribution guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
