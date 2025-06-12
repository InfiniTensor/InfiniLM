use super::{BatchStrategy, Req, Round, SessionId, SessionStub};
use log::warn;
use std::{cmp::min, collections::BTreeMap, iter::repeat_n, mem::take};

pub(crate) struct DefaultStrategy<T> {
    sess: BTreeMap<SessionId, SessionStub<T>>,
    pre_output: BTreeMap<SessionId, usize>,
    // 每次prefill的最大长度
    chunked_prefill_max_len: Option<usize>,
    max_toks: usize,
}

impl<T> DefaultStrategy<T> {
    pub fn new(chunked_prefill_len: Option<usize>, max_toks: usize) -> Self {
        Self {
            sess: Default::default(),
            pre_output: Default::default(),
            chunked_prefill_max_len: chunked_prefill_len,
            max_toks,
        }
    }
}

impl<T: 'static + Clone> BatchStrategy<T> for DefaultStrategy<T> {
    fn is_empty(&self) -> bool {
        self.sess.is_empty()
    }

    fn insert(&mut self, stub: SessionStub<T>) {
        assert!(self.sess.insert(stub.session.id, stub).is_none())
    }

    fn remove(&mut self, id: &SessionId) -> Option<SessionStub<T>> {
        self.sess.remove(id)
    }

    fn prepare(&mut self) -> Round<T> {
        let mut ans = Round::default();
        let mut out_idx = 0;

        let pre_output = take(&mut self.pre_output);

        let mut write_back_sessions = BTreeMap::new();

        while let Some((id, mut stub)) = self.sess.pop_first() {
            let max = stub.session.cache.capacity;
            let pos = stub.session.cache.len;
            let mut seq = stub.state.seq;
            let mut out = stub.state.out;
            let mut end = pos + seq;
            assert_eq!(out, 1, "TODO: 投机采样");
            //验证缓存是否溢出
            if end > max {
                warn!("cache overflow {end} > {max}");
                // 缓存溢出，不再推理
                ans.overflow.push(stub.session);
                continue;
            }

            // 用于限制每次tokens总数
            let remain_tok_num = self.max_toks - ans.tokens.len();
            assert!(remain_tok_num > 0);

            if let Some(prompt) = &stub.prompt {
                seq = self
                    .chunked_prefill_max_len
                    .map_or(min(remain_tok_num, seq), |chunked_prefill_max_len| {
                        remain_tok_num.min(seq).min(chunked_prefill_max_len)
                    });

                if seq < stub.state.seq {
                    // chunked prefill
                    out = 0;
                    end = pos + seq;

                    ans.tokens
                        .extend(prompt.iter().skip(prompt.len() - stub.state.seq).take(seq));

                    //更新stub信息
                    stub.state.seq -= seq;
                } else {
                    // 正常prefill
                    if seq != prompt.len() {
                        log::debug!("{:?} chunked prefil finished", id);
                    }
                    ans.tokens.extend(prompt[prompt.len() - seq..].to_owned());

                    stub.state.seq = 1;
                    stub.prompt = None;
                }
            } else {
                // decode
                assert_eq!(seq, 1);
                // fast embd
                ans.fast_map
                    .push((pre_output[&id] as _, ans.tokens.len() as _));
                ans.tokens.push(0)
            }

            // 尝试填充缓存
            stub.session.cache.len = end;
            // 填充推理信息
            ans.sample.extend(repeat_n(stub.session.sample_args, out));
            ans.output.push((id, out));
            ans.reqs.push(Req {
                cache: stub.session.cache.cache.clone(),
                pos,
                seq,
            });

            //输出处理
            //不会溢出 因为 out <= 1
            stub.state.remain_steps -= out;
            if stub.state.remain_steps == 0 {
                // 生成结束
                ans.finished.push(stub.session)
            } else {
                // 回填
                assert!(write_back_sessions.insert(id, stub).is_none());
                if out != 0 {
                    assert!(self.pre_output.insert(id, out_idx).is_none());
                }
            }
            out_idx += out;

            // 如果剩余tokens总数等于0，则退出循环
            if self.max_toks == ans.tokens.len() {
                break;
            }
        }
        self.sess.append(&mut write_back_sessions);
        ans
    }

    fn take_stubs(&mut self) -> Vec<SessionStub<T>> {
        take(&mut self.sess).into_values().collect()
    }
}
