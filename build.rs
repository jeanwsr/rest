extern crate dunce;
use std::env;
use std::path::PathBuf;

fn main() {

    // conditionally link to the libraries based on the features
    #[cfg(feature = "intel-mkl")] {
        println!("cargo:rustc-link-lib=mkl_rt");
        let blas_dir = if let Ok(blas_dir) = env::var("MKLROOT") {
            PathBuf::from(blas_dir)
        } else {panic!("MKLROOT not set for feature intel-mkl")};
        println!("cargo:rustc-link-search={}/lib",&blas_dir.display());
    }

    // AJZ34: After dftd3/4 v0.2, libxc v0.1, by default we support dynamic loading.
    // libs-dftd3.so, libdftd4.so, libxc.so are not necessarily linked at compile time.
    // User should specify those libraries if dftd3/4/libxc computation requested (by conda or LD_LIBRARY_PATH).
    //
    // #[cfg(feature = "dftd3")]
    // println!("cargo:rustc-link-lib=s-dftd3");
    // #[cfg(feature = "dftd4")]
    // println!("cargo:rustc-link-lib=dftd4");

    let library_names = ["cint","restmatr","hdf5","hdf5_shim","rest2fch","openblas","gomp"];
    library_names.iter().for_each(|name| {
        println!("cargo:rustc-link-lib={}",*name);
    });

    let external_dir = if let Ok(external_dir) = env::var("REST_EXT_DIR") {
        PathBuf::from(external_dir)
    } else {panic!("REST_EXT_DIR not set, which,however, is necessary for the build script to run.")};


    let library_path = [
        dunce::canonicalize(&external_dir).unwrap(),
    ];
    library_path.iter().for_each(|path| {
        println!("cargo:rustc-link-search={}",env::join_paths(&[path]).unwrap().to_str().unwrap())
    });
    //println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-lgomp");
}
