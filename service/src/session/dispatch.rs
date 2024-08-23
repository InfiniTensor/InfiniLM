use super::{batcher::Batcher, cache::Cache, task::Task};
use crate::ServiceComponent;
use causal_lm::{CausalLM, DecodingMeta, SampleArgs, SampleMeta};
use common::utok;
use std::{
    iter::zip,
    mem::{replace, size_of},
    str,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

pub(super) struct TaskHandle<M: CausalLM> {
    receiver: Option<UnboundedReceiver<utok>>,
    cache: Arc<Mutex<Option<Cache<M::Storage>>>>,
    buffer: Utf8Buffer,
}

impl<M: CausalLM> TaskHandle<M> {
    #[inline]
    pub fn take(&mut self) -> Cache<M::Storage> {
        // 停止响应接收
        let _ = self.receiver.take();
        // 取走 cache
        self.cache.lock().unwrap().take().unwrap()
    }
}

impl<M: CausalLM> ServiceComponent<M> {
    pub(super) fn infer(&self, sample: SampleArgs, mut cache: Cache<M::Storage>) -> TaskHandle<M> {
        let max = self.handle.model.max_seq_len() as usize;
        cache.reset_within_start_and_end_range(max / 4, max / 4, max / 4 * 3);
        // 生成推理任务与会话的交互管道
        let cache = Arc::new(Mutex::new(Some(cache)));
        let (sender, receiver) = unbounded_channel();
        self.handle
            .batcher
            .enq(Task::new(cache.clone(), sample, sender));
        TaskHandle {
            receiver: Some(receiver),
            cache,
            buffer: Default::default(),
        }
    }

    pub(super) async fn decode(&self, x: &mut TaskHandle<M>) -> Option<String> {
        loop {
            let s = x.receiver.as_mut().unwrap().recv().await.map(|token| {
                // detokenize and denormalize the token
                let ServiceComponent {
                    normalizer,
                    tokenizer,
                    ..
                } = self;
                normalizer.decode(tokenizer.decode(token))
            })?;
            let s = x.buffer.push(s.as_bytes());
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
}

pub(crate) struct Dispatcher<M: CausalLM> {
    pub model: M,
    pub(super) batcher: Arc<Batcher<Task<M::Storage>>>,
}

impl<M: CausalLM> From<M> for Dispatcher<M> {
    #[inline]
    fn from(model: M) -> Self {
        Self {
            model,
            batcher: Arc::new(Batcher::new()),
        }
    }
}

impl<M: CausalLM> Dispatcher<M> {
    /// 通过关闭任务队列通知推理线程退出。
    #[inline]
    pub fn stop(&self) {
        self.batcher.shutdown();
    }
}

impl<M> Dispatcher<M>
where
    M: CausalLM + 'static,
    M::Storage: Send,
{
    pub fn run(&self) {
        while let Some(tasks) = Some(self.batcher.deq()).filter(|t| !t.is_empty()) {
            // 锁定所有请求的缓存
            let mut caches = tasks.iter().map(Task::lock_cache).collect::<Vec<_>>();
            // 统计每个任务的查询长度
            let num_query = caches
                .iter()
                .map(|c| c.as_ref().map_or(0, |c| c.query().len()))
                .collect::<Vec<_>>();
            if num_query.iter().all(|&n| n == 0) {
                continue;
            }
            // 词嵌入
            let queries = caches
                .iter()
                .filter_map(|c| c.as_ref().map(Cache::query).filter(|q| !q.is_empty()))
                .flatten()
                .copied();
            let token_embedded = self.model.token_embed(queries);
            // 推理
            let queries = caches
                .iter_mut()
                .filter_map(|c| c.as_mut().map(Cache::as_ctx).filter(|q| q.seq_len() > 0));
            let hidden_state = self.model.forward(queries, token_embedded);
            drop(caches);
            // 采样
            let num_decode = tasks
                .iter()
                .map(|t| if t.is_alive() { 1 } else { 0 })
                .collect::<Vec<_>>();
            let decoding =
                zip(num_query, &num_decode).map(|(num_query, &num_decode)| DecodingMeta {
                    num_query,
                    num_decode,
                });
            let logits = self.model.decode(decoding, hidden_state);
            // 采样
            let args = zip(&tasks, &num_decode).map(|(t, &num_decode)| SampleMeta {
                num_decode,
                args: *t.sample(),
            });
            let tokens = self.model.sample(args, logits);
            // 为每次推理启动一个任务执行发射
            let eos = self.model.eos_token();
            let max = self.model.max_seq_len() as usize;
            let batcher = self.batcher.clone();
            tokio::task::spawn_blocking(move || {
                let end_size = max / 4;
                let start_size = max / 4;
                zip(tasks, num_decode)
                    .filter(|(_, n)| *n > 0)
                    .map(|(t, _)| t)
                    .zip(tokens)
                    .filter(|(_, token)| *token != eos)
                    .for_each(|(mut task, token)| {
                        if task.push(token, start_size, end_size, max) {
                            batcher.enq(task);
                        }
                    });
            });
        }
    }
}

#[derive(Clone, Default, Debug)]
struct Utf8Buffer(Vec<u8>);

impl Utf8Buffer {
    fn push(&mut self, bytes: impl AsRef<[u8]>) -> String {
        self.0.extend_from_slice(bytes.as_ref());
        let mut len = match str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(e) => e.valid_up_to(),
        };
        while len + size_of::<char>() <= self.0.len() {
            len += 1;
            match str::from_utf8(&self.0[len..]) {
                Ok(s) => len += s.as_bytes().len(),
                Err(e) => len += e.valid_up_to(),
            }
        }
        let s = self.0.split_off(len);
        let s = replace(&mut self.0, s);
        unsafe { String::from_utf8_unchecked(s) }
    }
}
