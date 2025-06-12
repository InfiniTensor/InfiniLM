use super::{Command, Output};
use crate::{
    CacheParts,
    batch::{BatchStrategy, DefaultStrategy, Round, SessionStub},
};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};

pub(super) struct EngineManager(DefaultStrategy<CacheParts>);

#[derive(Clone, Copy, Debug)]
pub enum CommandError {
    ShutDown,
    SendError,
    ReceiveError,
}

type E = CommandError;

impl EngineManager {
    pub fn new(chunked_prefill_len: Option<usize>, max_toks: usize) -> Self {
        Self(DefaultStrategy::new(chunked_prefill_len, max_toks))
    }

    /// 接收命令
    pub fn receive(
        &mut self,
        commands: &Receiver<Command>,
        outputs: &Sender<Output>,
    ) -> Result<(), E> {
        loop {
            // 总是尝试进行非阻塞接收
            loop {
                match commands.try_recv() {
                    Ok(cmd) => self.apply(cmd, outputs)?,
                    Err(TryRecvError::Disconnected) => return Err(E::ReceiveError),
                    Err(TryRecvError::Empty) => break,
                }
            }
            // 没有待处理的命令
            if self.0.is_empty() {
                // 也没有待处理的任务，阻塞等待
                match commands.recv() {
                    Ok(cmd) => self.apply(cmd, outputs)?,
                    Err(_) => break Err(E::ReceiveError),
                }
            } else {
                // 有待处理的任务，退出循环
                break Ok(());
            }
        }
    }

    /// 准备推理
    pub fn prepare(&mut self) -> Round<CacheParts> {
        self.0.prepare()
    }

    pub fn into_stubs(mut self) -> Vec<SessionStub<CacheParts>> {
        self.0.take_stubs()
    }

    fn apply(&mut self, cmd: Command, outputs: &Sender<Output>) -> Result<(), CommandError> {
        match cmd {
            Command::ShutDown => Err(CommandError::ShutDown),
            Command::Insert(req) => {
                self.0.insert(req.into_stub());
                Ok(())
            }
            Command::Remove(id) => {
                if self
                    .0
                    .remove(&id)
                    .is_none_or(|stub| outputs.send(Output::Removed(stub.session)).is_ok())
                {
                    Ok(())
                } else {
                    Err(CommandError::SendError)
                }
            }
        }
    }
}
