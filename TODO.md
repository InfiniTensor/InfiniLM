1. 目标：

    （miniCPM + Fastcache） x (DCU & 摩尔)
    （llava + Fastcache） x (DCU & 摩尔)


2. 具体工作拆分：
    
    a. DCU平台端到端跑通：llava encoder部分正确性调试【目前可上手的工作，缺的算子暂时先占位】 + 把encoder/Fastcache/llm拼到一起。
    
    b. 我今天搞：摩尔平台计算资源申请

    c. 我最近两天：两个平台缺的算子搞定

3. ddl: 10天后，本月25号

4. weight：
108：/home/weight/MiniCPM-V-2_6；/home/weight/llava-1.5-7b-hf