import subprocess
import os

# 定义可执行文件路径和输出文件路径
executable_path = "./build/test_debug"  # 根据你的编译路径调整
output_file = "debug_test_output.json"

# 调用编译后的程序
try:
    print("运行测试程序...")
    result = subprocess.run([executable_path], capture_output=True, text=True, check=True)
    print("程序输出:")
    print(result.stdout)
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print("程序运行失败:")
    print(e.stderr)
    exit(1)

# 检查输出文件是否生成
if os.path.exists(output_file):
    print(f"测试成功，生成的文件: {output_file}")
    # 如果需要，可以读取并打印 JSON 文件内容
    with open(output_file, "r", encoding="utf-8") as f:
        print("JSON 文件内容:")
        print(f.read())
else:
    print("测试失败，未找到输出文件。")
