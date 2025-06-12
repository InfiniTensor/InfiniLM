use llama_cu::{Cache, CacheParts, Session, SessionId, TextBuf};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

pub(super) struct AppSession {
    name: String,
    msgs: Vec<String>,
    info: Option<Session<CacheParts>>,
    pub buf: TextBuf,
}

impl AppSession {
    pub fn new(name: impl ToString, cache: Cache<CacheParts>) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(0);
        let session_id = SessionId(ID.fetch_add(1, SeqCst));
        Self {
            name: format!("{} {}", name.to_string(), session_id.0),
            msgs: vec![String::new()],
            info: Some(Session {
                id: session_id,
                sample_args: Default::default(),
                cache,
            }),
            buf: TextBuf::new(),
        }
    }

    pub fn id(&self) -> SessionId {
        self.info.as_ref().unwrap().id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn msgs(&self) -> &[String] {
        &self.msgs
    }

    pub fn last_sentence_mut(&mut self) -> &mut String {
        self.msgs.last_mut().unwrap()
    }

    pub fn start(&mut self) -> Option<(Session<CacheParts>, String)> {
        let ans = self
            .info
            .take()
            .map(|s| (s, self.msgs.last().unwrap().clone()));
        self.msgs.push(Default::default());
        ans
    }

    pub fn idle(&mut self, session: Session<CacheParts>) {
        self.msgs.push(Default::default());
        assert!(self.info.replace(session).is_none())
    }

    pub fn is_busy(&self) -> bool {
        self.info.is_none()
    }
}
