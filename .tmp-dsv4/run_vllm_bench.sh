#!/usr/bin/env bash
set -u
cd /home/wuwei/InfiniLM/.tmp-dsv4
echo "VLLM_DRIVER_START $(date -Is)"
python3 bench_latency.py --url http://127.0.0.1:18082 --engine vllm --model /home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --conc 4 --input 1024 --output 1024 --timeout 21600
python3 bench_latency.py --url http://127.0.0.1:18082 --engine vllm --model /home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --conc 4 --input 4096 --output 4096 --timeout 21600
python3 bench_latency.py --url http://127.0.0.1:18082 --engine vllm --model /home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --conc 16 --input 128 --output 128 --timeout 21600
python3 bench_latency.py --url http://127.0.0.1:18082 --engine vllm --model /home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --conc 16 --input 1024 --output 1024 --timeout 21600
python3 bench_latency.py --url http://127.0.0.1:18082 --engine vllm --model /home/wuwei/models/Eco-Tech_DeepSeek-V4-Flash-w8a8-mtp --conc 16 --input 4096 --output 4096 --timeout 21600
echo "VLLM_DRIVER_DONE $(date -Is)"