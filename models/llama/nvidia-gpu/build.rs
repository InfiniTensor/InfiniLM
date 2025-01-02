fn main() {
    use build_script_cfg::Cfg;
    use search_corex_tools::find_corex;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};

    let nvidia = Cfg::new("use_nvidia");
    let iluvatar = Cfg::new("use_iluvatar");
    let nccl = Cfg::new("use_nccl");

    let nvidia_detected = find_cuda_root().is_some();
    let iluvatar_detected = find_corex().is_some();

    if nvidia_detected {
        nvidia.define();
        if find_nccl_root().is_some() {
            nccl.define()
        }
    }

    if iluvatar_detected {
        iluvatar.define()
    }
}
