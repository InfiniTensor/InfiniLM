use std::env;
use std::path::PathBuf;

fn main() {
    use build_script_cfg::Cfg;
    use search_neuware_tools::find_neuware_home;

    let neuware = Cfg::new("detected_neuware");
    if find_neuware_home().is_some() {
        neuware.define();
    }

    // Tell cargo to tell rustc to link the shared library.
    println!("cargo:rustc-link-search=native=/home/duanchenjie/workspace/operators/build/linux/x86_64/release");
    // 链接动态库，不要包含前缀 lib 和后缀 .so
    println!("cargo:rustc-link-lib=dylib=operators"); // 动态库名为 liboperators.so
    // Link the OpenMP library
    println!("cargo:rustc-link-lib=dylib=gomp");    

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })     
        // Tell cargo to invalidate the built crate whenever the wrapper changes
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Disable layout tests because bitfields might cause issues
        .layout_tests(false)
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");      
}
