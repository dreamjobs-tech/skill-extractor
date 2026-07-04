fn main() {
    // ort-sys's prebuilt macOS binaries compile the CoreML execution provider
    // into the main static lib but only emit the framework link directive when
    // a separate providers lib exists — which it doesn't in this distribution.
    // Without this, linking fails with undefined _OBJC_CLASS_$_MLComputePlan.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=framework=CoreML");
    }
}
