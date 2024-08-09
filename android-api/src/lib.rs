use service::Service;
use jni::JNIEnv;
use jni::JavaVM;
use jni::objects::{JObject, JString,JValue,GlobalRef};
use jni::sys::{jint, jstring};
use tokio::runtime::Runtime;
use tokio::task::JoinHandle;
use tokio;
use llama_cpu::Transformer as M;
use android_logger::Config;
use log::info;
use std::{thread::spawn, path::Path, time::Instant};
use std::sync::{Arc, Mutex};
use once_cell::sync::OnceCell;  // 使用 OnceCell 管理静态模型
use lazy_static::lazy_static;
use once_cell::sync::Lazy;

static MODEL_INSTANCE: OnceCell<Service<M>> = OnceCell::new();
// 初始化一个全局的 Tokio 运行时
static RUNTIME: Lazy<Mutex<Runtime>> = Lazy::new(|| {
    Mutex::new(
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime"),
    )
});

#[no_mangle]
pub extern "system" fn Java_com_example_helloworld_MainActivity_initializeLogger() {
    android_logger::init_once(
        Config::default().with_min_level(log::Level::Info)
    );
}
#[no_mangle]
pub extern "system" fn Java_com_example_helloworld_MainActivity_initialize(
    env: JNIEnv<'_>,
    _class: JObject<'_>,
    model_path: JString,
) {
    info!("Initializing model");
    let model_path: String = env.get_string(model_path).expect("Invalid model path").into();
    info!("Model path: {}", model_path);
    let runtime = RUNTIME.lock().unwrap();
    runtime.block_on(async {
        MODEL_INSTANCE.get_or_init(|| {
            info!("Initializing model...");
            // 模型加载必须是异步的
            let (service, _handle) = Service::<M>::load(&model_path, ());
            info!("Model initialized successfully.");
            service
        });
    });
    info!("Model initialization completed.");
}

#[no_mangle]
pub extern "system" fn Java_com_example_helloworld_MainActivity_generate(
    env: JNIEnv<'_>,
    _class: JObject<'_>,
    prompt: JString,
    max_steps: jint,
    callback: JObject<'_>,
){
    info!("Starting generate function");
    let prompt: String = env.get_string(prompt).expect("Invalid prompt").into();
    info!("Prompt: {}", prompt);
    let max_steps = max_steps as usize;
    info!("Max steps: {}", max_steps);
    let rt = RUNTIME.lock().unwrap();
    rt.block_on(async {
        generate_text(env, prompt, max_steps, callback).await;
    });
}

pub async fn generate_text(
    env: JNIEnv<'_>,
    prompt: String,
    max_steps: usize,
    callback: JObject<'_>,
){
    let prompt = if Path::new(&prompt).is_file() {
        println!("prompt from file: {}", prompt);
        std::fs::read_to_string(&prompt).unwrap()
    } else {
        prompt
    };
    println!("Prompt read: {}", prompt);
    info!("Prompt read: {}", prompt);
    if let Some(service) = MODEL_INSTANCE.get() {
        let mut generator = service.generate(&*prompt, None);
        let mut steps = 0;
        let time = Instant::now();
        while let Some(s) = generator.decode().await {
            let text_chunk = match &*s {
                "\\n" => "\n".to_string(),
                _ => s,
            };
            let java_string = env.new_string(text_chunk).unwrap();
            env.call_method(
                callback,
                "onTextGenerated",
                "(Ljava/lang/String;)V",
                &[JValue::from(java_string)], 
            ).unwrap();
            steps += 1;
            if steps == max_steps {
                break;
            }
        }
        let time = time.elapsed();
        println!("Time elapsed: {:?}/tok", time.div_f32(steps as f32));
        info!("Time elapsed: {:?}/tok", time.div_f32(steps as f32));
        println!("Generation complete with {} steps", steps);
    }else {
        println!("Model not initialized!");
    }
}