我需要在dump_kv下提供了一个测试结果， input_kv.bin是输入的kvcache
output_kv.bin是输出的正确结果。
 输出内容

  - dump_kv/input_kv.bin 与 dump_kv/output_kv.bin
  - dump_kv/meta.json 含二进制索引与元信息：
      - binary_index.input / binary_index.output: 列出每个块的
          - name: k_l00 / v_l00 / ...
          - shape: [B, H, S, D]
          - offset_elems: 在对应 .bin 文件中的元素偏移（非字节）
          - n_elems: 元素数
  - 内存布局：行主序（row-major）[B,H,S,D]
  - 写入顺序：每层 K 紧接 V，依次 layer 0..L-1
  - 重建单层偏移（字节）:
      - byte_offset = offset_elems * sizeof(dtype)
      - 读取 n_elems 元素并按 shape 还原