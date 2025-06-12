use llama_cu::{DistKVCache, SampleArgs, Session, SessionId, Terminal, utok};
use std::{
    collections::BTreeMap,
    iter::zip,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
    time::Instant,
};

pub(crate) struct CacheManager {
    terminal: Terminal,
    caches: BTreeMap<Instant, (Vec<utok>, DistKVCache)>,
}

impl CacheManager {
    pub fn new(terminal: Terminal) -> Self {
        Self {
            terminal,
            caches: Default::default(),
        }
    }

    pub fn send(
        &mut self,
        tokens: Vec<utok>,
        sample_args: SampleArgs,
        max_tokens: usize,
    ) -> (SessionId, Vec<utok>) {
        static SESSION_ID: AtomicUsize = AtomicUsize::new(0);
        let id = SessionId(SESSION_ID.fetch_add(1, SeqCst));

        let use_cache = &tokens[..tokens.len() - 1];
        let best_cache = self
            .caches
            .iter()
            .map(|(key, (history, _))| (*key, common_len(history, use_cache)))
            .max_by_key(|&(_, len)| len);

        let cache = match best_cache {
            Some((key, pos)) => {
                let (_, mut cache) = self.caches.remove(&key).unwrap();
                cache.pos = pos;
                cache
            }
            None => self.terminal.new_cache(),
        };
        let pos = cache.pos;
        self.terminal.start(
            Session {
                id,
                sample_args,
                cache,
            },
            &tokens[pos..],
            max_tokens,
        );
        (id, tokens)
    }

    pub fn insert(&mut self, tokens: Vec<utok>, cache: DistKVCache) {
        assert!(
            self.caches
                .insert(Instant::now(), (tokens, cache))
                .is_none()
        )
    }
}

fn common_len<T: Eq>(a: &[T], b: &[T]) -> usize {
    zip(a, b).take_while(|(a, b)| a == b).count()
}
