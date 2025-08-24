# InfiniCore 开发者手册

Dear 开发者，感谢你参与 InfiniCore 开源项目的开发！本文档将帮助你了解如何向 InfiniCore 项目贡献代码。

## 项目介绍

### 项目模块体系

- infini-utils：全模块通用工具代码。
- infinirt：运行时库，依赖 infini-utils。
- infiniop：算子库，依赖 infinirt。除了 C++ 算子实现之外，也包括使用九齿（triton）的算子实现，这部分算子需要在编译之前使用脚本生成源文件。安装后可以运行位于 `test/infiniop` 中的单测脚本进行测试。
- infiniccl：通信库，依赖 infinirt。
- utils-test：工具库测试代码，依赖 infini-utils。
- infiniop-test：算子库测试框架代码。与单测不同，读取gguf测例文件进行测试（详见[`测例文档`](test/infiniop-test/README.md)）。使用前需要安装好 infiniop。
- infiniccl-test：通信库测试代码，使用前需要安装好 infiniccl。

### 文件目录结构

```bash
├── xmake.lua  # 总体 xmake 编译配置，包含所有平台的编译选项和宏定义
├── xmake/*.lua  # 各平台 xmake 编译配置， 包含各平台特有的编译方式
│    
├── include/  # 对外暴露的头文件目录，安装时会被复制到安装目录
│   ├── infiniop/*.h  # InfiniOP算子库子头文件
│   ├── *.h  # 模块核心头文件
│ 
├── src/  # 各模块源代码目录，包含源代码文件以及不对外暴露的头文件
│   ├── infiniop/ # InfiniOP算子库源代码目录
│   │   ├── devices/  # 每个设备平台各自的通用代码目录
│   │   ├── ops/ # 算子实现代码目录
│   │   │   ├── [op]/
│   │   │   │   ├── [device]/ # 各硬件平台算子实现代码目录
│   │   │   │   ├── operator.cc # 算子C语言接口实现
│   │   ├── reduce/ # 规约类算子通用代码目录
│   │   ├── elementwise/  # 逐元素类算子通用代码目录
│   │   ├── *.h  # 核心结构体定义
│   │
│   ├── infiniop-test/  # InfiniOP算子库测试框架
│   ├── infinirt/ # InfiniRT运行时库源代码目录
│   ├── infiniccl/ # InfiniCCL集合通信库源代码目录
│  
├── test/ # 测试源代码目录
│   ├── infiniop/ # InfiniOP算子库单元测试目录
│   │       ├── *.py     # 单测脚本（依赖各平台PyTorch）
│   ├── infiniop-test/
│   │       ├── test_generate/ # 算子库测试框架测例生成脚本
│  
├── scripts/ # 脚本目录
│   ├── install.py # 安装编译脚本
│   ├── python_test.py # 运行所有单测脚本
```

## 开发引导

### 代码提交流程

1. 在github仓库issue页面根据任务类型（开发或bug）创建 issue，所有commit必须有对应的 issue 编号。
2. 外部人员需要通过 fork 代码仓库提交 PR。
3. 根据 issue 编号建立分支，分支名字格式为 `issue/#` （# 为issue 编号）。如果出现重复，可在后面添加“-#”序号，或用“/”后增加说明。
4. 所有 commit 信息必须以 `issue/#` 开头，
5. 分支推到远程后，建 Pull Request，标题需要以 `issue/#` 开头。在原issue页面上将该PR关联。
6. PR必须添加至少两位审核员（模块负责人和项目管理员等），PR中需附上最后一次修改后测试通过的截图。
7. PR通过审核，通过自动测试，无代码冲突后方可合并。合并后，关闭原 issue。

### 如何开发一个新算子

1. 根据算子定义设计算子接口，在 [`InfiniCore文档`](https://github.com/InfiniTensor/InfiniCore-Documentation) 中添加算子文档。提交文档 PR 。
2. 在 `include/infiniop/` 中添加算子头文件，并 include 到 `include/infiniop.h` 中。
3. 在 `src/infiniop/ops/` 中添加算子实现目录，并在目录中创建 `operator.cc` 文件实现头文件中的接口。
4. 在 `src/infiniop/ops/[op]/[device]/` 中添加平台算子实现。注意复用平台公共代码（比如逐元素计算和规约计算），开发过程中把未来可复用的代码写在相应公用代码目录里。比如 cuda kernel 可以多个平台公用，可以考虑在头文件中实现，并在多个源文件中使用。
5. 算子实现可以成功编译安装后，在 `test/infiniop/` 中添加单测脚本，与 PyTorch 实现进行正确性和性能比较。测例应覆盖算子常用类型和形状。测试成功之后可以将测例添加至 `scripts/python_test.py` 一键测试脚本中（这样 Github 自动测试也会包含该算子）。
6. 在 `test/infiniop-test/` 算子测试框架中添加该算子的测例脚本。脚本应该包含构建该算子 gguf 测例的类，并在 main 函数中添加几个随机测例。验证随机 gguf 测例可以通过测试框架的测试程序。
7. 按照流程提交代码 PR 。

### C++ 代码命名书写规范

1. 类型

    内部数据结构类型 `UpperCamelCase`

    ```c++
    // 尽量使用 Infinixx 开头
    struct InfiniopMatmulCudaDescriptor;
    
    template <typename KeyType, typename ValueType>
    class HashMap; 
    
    using ValueMap = std::unordered_map<int, std::string>;
    ```

    对外暴露的指针类型和枚举类型 `infinixx[XxxXxx]_t`

    常量使用 `INFINI_UPPER_SNAKE_CASE`

    ```c++
    typedef struct InfiniopMatmulCudaDescriptor *infiniopMatmulCudaDescriptor_t;
    
    typedef enum {
        // INFINI...
        INFINI_DTYPE_INVALID = 0,
    } infiniDtype_t;
    ```

2. 普通变量、形参、类数据成员，使用 `snake_case`

    成员名前下划线特指private成员，其他情况应避免使用前下划线

    ```c++
    int max_count;
    
    class Example {
    public:
        std::string getUserName(std::string user_id);
    private:
        // private数据成员名字前加下划线
        int _max_count;
        std::string _user_name;
    };
    
    struct UrlTableProperties {  
        string name;
        int num_entries;  
        static Pool<UrlTableProperties>* pool;
    };
    ```

    当形参与函数内部变量或成员变量重名，可选择其中一个名字后加下划线。当函数内部临时变量和成员重名时，临时变量名字后加下划线。后下划线表示“临时”

    ```c++
    void do(int count_){
        int count = count_;
    }
    ```

3. 函数，使用 lowerCamelCase

    ```c++
    int getMaxValue() const;
    ```

4. const/volatile修饰符写在类型前面

    ```c++
    const void *ptr;
    const int num;
    ```

### 代码格式化

本项目分别使用 `clang-format-16` 和 `black` 对 C/C++ 以及 Python 代码进行格式化。可以使用 [`scripts/format.py`](/scripts/format.py) 脚本实现代码格式化检查和操作。

使用

```shell
python scripts/format.py -h
```

查看脚本帮助信息：

```plaintext
usage: format.py [-h] [--ref REF] [--path [PATH ...]] [--check] [--c C] [--py PY]

options:
  -h, --help         show this help message and exit
  --ref REF          Git reference (commit hash) to compare against.
  --path [PATH ...]  Files to format or check.
  --check            Check files without modifying them.
  --c C              C formatter (default: clang-format-16)
  --py PY            Python formatter (default: black)
```

参数中：

- `ref` 和 `path` 控制格式化的文件范围
  - 若 `ref` 和 `path` 都为空，格式化当前暂存（git added）的文件；
  - 否则
    - 若 `ref` 非空，将比较指定 commit 和当前代码的差异，只格式化修改过的文件；
    - 若 `path` 非空，可传入多个路径（`--path p0 p1 p2`），只格式化指定路径及其子目录中的文件；
- 若设置 `--check`，将检查代码是否需要修改格式，不修改文件内容；
- 通过 `--c` 指定 c/c++ 格式化器，默认为 `clang-format-16`；
- 通过 `--python` 指定 python 格式化器 `black`；

### vscode 开发配置

基本配置见 [xmake 官方文档](https://xmake.io/#/zh-cn/plugin/more_plugins?id=%e9%85%8d%e7%bd%ae-intellsence)。

- TL;DR
  - clangd

    打开 *xmake.lua*，保存一次以触发编译命令生成，将在工作路径下自动生成 *.vscode/compile_commands.json* 文件。然后在这个文件夹下创建 *settings.json*，填入：

    > .vscode/settings.json

    ```json
    {
        "clangd.arguments": [
            "--compile-commands-dir=.vscode"
        ],
        "xmake.additionalConfigArguments": [
            // 在这里配置 XMAKE_CONFIG_FLAGS
            "--nv-gpu=y"
        ],
    }
    ```
