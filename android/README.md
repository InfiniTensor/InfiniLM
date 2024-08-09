# 安卓对话支持

使用 [cargo-ndk](https://crates.io/crates/cargo-ndk) 编译：

```shell
cargo ndk -t armeabi-v7a -t arm64-v8a -t x86_64 -o target/jniLibs build --package android --release
```

> **NOTICE** 需要按安装 cargo-ndk 并按文档描述配置 NDK 和安卓工具链。

> **NOTICE** 如果需要支持的目标硬件较少，可以减少 `-t xxx` 选项。

> **NOTICE** 必须使用 `--release` 编译。gemm simd 内联汇编在 debug 模式下无法编译。

将在 target 目录生成 jniLibs 目录，将这个目录拷贝到安卓项目的 *app/src/main* 目录中。并添 Native.java：

```java
package org.infinitensor.lm;

public class Native {
    // 加载模型并启动推理服务，必须最先调用。
    public native static void init(String model_path);
    // 开始对话。
    public native static void start(String prompt);
    // 终止对话。
    public native static void abort();
    // 解码模型反馈。
    public native static String decode();
}
```

调用这些代码后：

```kotlin
try {
    System.loadLibrary("infinilm_chat")
} catch (e: UnsatisfiedLinkError) {
    throw RuntimeException("Native library not found", e)
}
```

即可使用 `Native` 类进行推理。
