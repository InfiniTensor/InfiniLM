use std::sync::{
    mpsc::{Receiver, Sender},
    Mutex, OnceLock,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

static COMMAND: OnceLock<UnboundedSender<(String, Sender<String>)>> = OnceLock::new();
static DIALOG: OnceLock<Mutex<Option<Receiver<String>>>> = OnceLock::new();

pub fn init() -> UnboundedReceiver<(String, Sender<String>)> {
    let (sender, receiver) = unbounded_channel();
    COMMAND.get_or_init(move || sender);
    DIALOG.get_or_init(Default::default);
    receiver
}

pub fn chat(prompt: String, sender: Sender<String>) {
    if let Some(command) = COMMAND.get() {
        command.send((prompt, sender)).unwrap();
    } else {
        log::error!("Command channel not initialized");
        panic!();
    };
}

pub fn dialog() -> &'static Mutex<Option<Receiver<String>>> {
    if let Some(dialog) = DIALOG.get() {
        dialog
    } else {
        log::error!("Dialog channel not initialized");
        panic!();
    }
}

/// 验证 tokio mpsc channel 不必在异步上下文中创建。
#[test]
fn test_behavior() {
    let (sender, mut receiver) = unbounded_channel();

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let handle = runtime.spawn(async move {
        receiver.recv().await.unwrap();
    });

    sender.send(()).unwrap();
    runtime.block_on(handle).unwrap();
    runtime.shutdown_background();
}
