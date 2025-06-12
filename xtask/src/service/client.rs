use super::V1_CHAT_COMPLETIONS;
use log::{info, trace, warn};
use openai_struct::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest, CreateChatCompletionStreamResponse,
};
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
use std::{env::VarError, time::Instant};
use tokio::time::Duration;
use tokio_stream::StreamExt;

const CONCURRENT_REQUESTS: usize = 10;

fn requset_body_chat(prompt: &str) -> String {
    serde_json::to_string(&CreateChatCompletionRequest {
        model: "model".into(),
        messages: vec![ChatCompletionRequestMessage::User(
            openai_struct::ChatCompletionRequestUserMessage {
                content: serde_json::Value::String(prompt.into()),
                name: None,
            },
        )],
        metadata: None,
        service_tier: None,
        audio: None,
        function_call: None,
        functions: None,
        max_completion_tokens: None,
        max_tokens: Some(256),
        modalities: None,
        n: None,
        parallel_tool_calls: None,
        prediction: None,
        reasoning_effort: None,
        response_format: None,
        store: None,
        tool_choice: None,
        tools: None,
        top_logprobs: None,
        web_search_options: None,
        frequency_penalty: None,
        logit_bias: None,
        logprobs: None,
        presence_penalty: None,
        seed: None,
        stop: None,
        stream: Some(true),
        stream_options: None,
        temperature: None,
        top_p: None,
        user: None,
    })
    .unwrap()
}

fn create_client_with_headers() -> (reqwest::Client, HeaderMap) {
    let client = reqwest::Client::new();
    let mut headers: HeaderMap = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    (client, headers)
}

async fn send_single_request(
    port: u16,
    client: &reqwest::Client,
    headers: &HeaderMap,
    req_body: String,
    index: Option<usize>,
) -> Result<(usize, usize, String, Duration), String> {
    let task_start = Instant::now();
    let index = index.unwrap_or(0);

    if index > 0 {
        trace!("任务 {} 开始", index);
    }

    let req = client
        .post(format!("http://localhost:{port}{V1_CHAT_COMPLETIONS}"))
        .headers(headers.clone())
        .body(req_body)
        .timeout(Duration::from_secs(100));

    match req.send().await {
        Ok(res) => {
            let status = res.status();
            if index > 0 {
                info!(
                    "任务 {} - 响应状态: {}, 耗时: {:?}",
                    index,
                    status,
                    task_start.elapsed()
                );
            } else {
                info!("响应状态: {}, header={:#?}", status, res.headers());
            }

            if status.is_success() {
                if index > 0 {
                    trace!("任务 {} 开始读取流式响应...", index);
                } else {
                    trace!("开始读取流式响应...");
                }

                let mut stream = res.bytes_stream();
                let mut chunk_count = 0;
                let mut accumulated_content = String::new();
                let mut buffer = String::new();

                while let Some(item) = stream.next().await {
                    chunk_count += 1;
                    match item {
                        Ok(bytes) => {
                            let text = std::str::from_utf8(&bytes).unwrap_or("<invalid utf8>");
                            buffer.push_str(text);

                            if index > 0 {
                                trace!(
                                    "任务 {} 收到第 {} 个数据块: {:?}",
                                    index, chunk_count, text
                                );
                            } else {
                                let now = Instant::now();
                                trace!("收到第 {} 个数据块 - 时间: {:?}", chunk_count, now);
                                trace!("原始数据: {:?}", text);
                            }

                            // 处理可能跨越多个数据块的SSE消息
                            while let Some(end_pos) = buffer.find("\n\n") {
                                let sse_chunk = buffer[..end_pos].to_string();
                                buffer.drain(..end_pos + 2);

                                // 解析 SSE 格式，以 data: 为准
                                for line in sse_chunk.lines() {
                                    if let Some(data_line) = line.strip_prefix("data: ") {
                                        // 尝试解析为CreateChatCompletionResponse
                                        match serde_json::from_str::<
                                            CreateChatCompletionStreamResponse,
                                        >(data_line)
                                        {
                                            Ok(response) => {
                                                trace!("解析响应: {response:?}");

                                                // 提取choices数组中的content
                                                for choice in &response.choices {
                                                    if let Some(content) = &choice.delta.content {
                                                        accumulated_content.push_str(content);
                                                        trace!("提取到文本内容: {:?}", content)
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                if index == 0 {
                                                    trace!(
                                                        "响应解析失败: {} - 原始数据: {:?}",
                                                        e, data_line
                                                    );
                                                }
                                                // 如果不是有效的响应格式，可能是纯文本内容
                                                accumulated_content.push_str(data_line);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!("任务 {} 读取流时出错: {:?}", index, e);
                            break;
                        }
                    }
                }

                if index > 0 {
                    info!(
                        "任务 {} 完成 - 总耗时: {:?}, 数据块数: {}, 内容长度: {}",
                        index,
                        task_start.elapsed(),
                        chunk_count,
                        accumulated_content.len()
                    );
                } else {
                    println!("流式响应结束，共收到 {} 个数据块", chunk_count);
                    println!("完整生成内容: {}", accumulated_content);
                }

                Ok((
                    index,
                    chunk_count,
                    accumulated_content,
                    task_start.elapsed(),
                ))
            } else {
                let error_text = res.text().await.unwrap_or_default();
                if index > 0 {
                    warn!(
                        "任务 {} 失败 - 状态: {}, 错误: {}",
                        index, status, error_text
                    );
                } else {
                    println!("body: {}", error_text);
                }
                Err(format!("HTTP错误: {}", status))
            }
        }
        Err(e) => {
            if index > 0 {
                warn!("任务 {} 请求失败: {:?}", index, e);
            }
            Err(format!("请求错误: {:?}", e))
        }
    }
}

#[test]
fn test_post_send() {
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(VarError::NotPresent) => return,
        Err(e) => panic!("{e:?}"),
    };

    crate::logger::init();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let (client, headers) = create_client_with_headers();

            info!(
                "Runtime workers = {}",
                tokio::runtime::Handle::current().metrics().num_workers()
            );

            let req_body = requset_body_chat("Tell a story");

            trace!("send req");
            let _ = send_single_request(port, &client, &headers, req_body, None).await;
        })
}

#[test]
fn test_post_send_multi() {
    let port = match std::env::var("TEST_PORT") {
        Ok(port) => port.parse().unwrap(),
        Err(VarError::NotPresent) => return,
        Err(e) => panic!("{e:?}"),
    };

    crate::logger::init();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(async move {
            let (client, headers) = create_client_with_headers();

            info!(
                "Runtime workers = {}",
                tokio::runtime::Handle::current().metrics().num_workers()
            );

            // 创建多个不同的请求内容
            let request_bodies = (0..CONCURRENT_REQUESTS)
                .map(|i| requset_body_chat(&format!("Tell me a story number {}", i + 1)))
                .collect::<Vec<_>>();

            let start_time = Instant::now();
            info!("开始发送 {} 个并发请求", CONCURRENT_REQUESTS);

            // 创建并发任务
            let tasks = request_bodies
                .into_iter()
                .enumerate()
                .map(|(index, req_body)| {
                    let client = client.clone();
                    let headers = headers.clone();

                    tokio::spawn(async move {
                        send_single_request(port, &client, &headers, req_body, Some(index + 1))
                            .await
                    })
                })
                .collect::<Vec<_>>();

            // 等待所有任务完成并统计结果
            let total_elapsed = start_time.elapsed();
            let mut successful_count = 0;
            let mut failed_count = 0;
            let mut total_chunks = 0;
            let mut total_text_length = 0;
            let mut max_duration = Duration::ZERO;
            let mut min_duration = Duration::MAX;

            for task in tasks {
                match task.await {
                    Ok(Ok((index, chunks, content, duration))) => {
                        successful_count += 1;
                        total_chunks += chunks;
                        total_text_length += content.len();
                        max_duration = max_duration.max(duration);
                        min_duration = min_duration.min(duration);
                        trace!("任务 {} 成功完成", index);
                    }
                    Ok(Err(e)) => {
                        failed_count += 1;
                        warn!("任务失败: {}", e);
                    }
                    Err(e) => {
                        failed_count += 1;
                        warn!("任务执行出错: {:?}", e);
                    }
                }
            }

            // 输出统计信息
            println!("\n=== 并发测试统计 ===");
            println!("总请求数: {}", CONCURRENT_REQUESTS);
            println!("成功请求数: {}", successful_count);
            println!("失败请求数: {}", failed_count);
            println!("总耗时: {:?}", total_elapsed);
            println!("最快请求: {:?}", min_duration);
            println!("最慢请求: {:?}", max_duration);
            println!(
                "平均每请求耗时: {:?}",
                total_elapsed / CONCURRENT_REQUESTS as u32
            );
            println!("总数据块数: {}", total_chunks);
            println!("总文本长度: {}", total_text_length);
            println!(
                "成功率: {:.1}%",
                (successful_count as f64 / CONCURRENT_REQUESTS as f64) * 100.0
            );

            if successful_count > 0 {
                println!(
                    "平均每请求数据块数: {:.1}",
                    total_chunks as f64 / successful_count as f64
                );
                println!(
                    "平均每请求文本长度: {:.1}",
                    total_text_length as f64 / successful_count as f64
                );
            }

            // 验证至少有一些请求成功
            assert!(successful_count > 0, "至少应该有一个请求成功");
        })
}
