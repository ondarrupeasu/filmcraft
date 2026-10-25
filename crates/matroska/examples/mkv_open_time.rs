//! How long opening (indexing) a Matroska file takes: `cargo run --release -p filmcraft-matroska
//! --example mkv_open_time -- FILE…`.

fn main() {
    if let Ok(dir) = std::env::var("MKV_INDEX_CACHE") {
        filmcraft_matroska::set_index_cache_dir(Some(dir.into()));
    }
    for path in std::env::args().skip(1) {
        let t0 = std::time::Instant::now();
        let f = std::fs::File::open(&path).expect("open");
        match filmcraft_matroska::open(f) {
            Ok(m) => {
                let n: usize = m.tracks.iter().map(|t| t.samples.len()).sum();
                println!("{:.2} s  {} samples, {} clusters  {path}", t0.elapsed().as_secs_f64(), n, m.clusters.len());
            }
            Err(e) => println!("error {e}  {path}"),
        }
    }
}
