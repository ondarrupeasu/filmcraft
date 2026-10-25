//! A file opened from the kept index is the same as indexed from scratch:
//! `MKV_INDEX_CACHE=dir cargo run --release -p filmcraft-matroska --example mkv_index_same -- FILE…`.

fn main() {
    let dir = std::env::var("MKV_INDEX_CACHE").expect("MKV_INDEX_CACHE");
    for path in std::env::args().skip(1) {
        filmcraft_matroska::set_index_cache_dir(None);
        let fresh = filmcraft_matroska::open(std::fs::File::open(&path).expect("open")).expect("fresh");
        filmcraft_matroska::set_index_cache_dir(Some(dir.clone().into()));
        let _ = filmcraft_matroska::open(std::fs::File::open(&path).expect("open")).expect("written");
        let kept = filmcraft_matroska::open(std::fs::File::open(&path).expect("open")).expect("kept");
        let same = format!("{fresh:?}") == format!("{kept:?}");
        println!("{}  {path}", if same { "same" } else { "DIFFERENT" });
        if !same {
            std::fs::write("/tmp/fresh.txt", format!("{fresh:#?}")).ok();
            std::fs::write("/tmp/kept.txt", format!("{kept:#?}")).ok();
        }
    }
}
