//! Sample tables kept on disk between opens (opt-in with [`set_index_cache_dir`]).
//!
//! Indexing reads every cluster of the file: a two-hour MKV is read end to end, seconds on a fast
//! disk. With a cache directory set, the tables built by a full open are written there and the next
//! open of the same file reads only its metadata (Info, Tracks, Cues…) and takes the tables from the
//! cache. A file is recognized by its length and a hash of three 64 KiB windows (start, middle,
//! end), so a rewritten file is indexed again.

use std::path::PathBuf;
use std::sync::Mutex;

use crate::meta::ClusterInfo;
use crate::source::ByteSource;
use crate::track::Sample;

static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Files smaller than this are indexed every time (it is quick); in tests, every file is kept.
const MIN_LEN: u64 = if cfg!(test) { 1 } else { 64 << 20 };
const MAGIC: &[u8; 8] = b"FCMKVIDX";
const VERSION: u32 = 1;
const WINDOW: usize = 64 * 1024;

/// Where indexes are kept (`None`: nowhere, every open indexes the file; the default).
pub fn set_index_cache_dir(dir: Option<PathBuf>) {
    *DIR.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = dir;
}

fn dir() -> Option<PathBuf> {
    DIR.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

/// What a full open found beyond the metadata: per track (by `TrackNumber`) its samples, the
/// clusters, the level-1 elements met among them, and the warnings.
pub(crate) struct Kept {
    pub clusters: Vec<ClusterInfo>,
    pub level1: Vec<(u32, u64)>,
    pub warnings: Vec<String>,
    pub tracks: Vec<(u64, Vec<Sample>)>,
}

/// The cache file for `src`, when caching is on and the file is worth it.
pub(crate) fn path_for<S: ByteSource + ?Sized>(src: &S) -> Option<PathBuf> {
    let dir = dir()?;
    let len = src.len();
    if len < MIN_LEN {
        return None;
    }
    let mut h = Fnv::new();
    h.u64(len);
    let mut buf = vec![0u8; WINDOW.min(usize::try_from(len).unwrap_or(WINDOW))];
    for at in [0, len / 2, len - buf.len() as u64] {
        let at = at.min(len - buf.len() as u64);
        src.read_at(at, &mut buf).ok()?;
        h.bytes(&buf);
    }
    Some(dir.join(format!("{:016x}.mkvidx", h.0)))
}

pub(crate) fn load(path: &std::path::Path) -> Option<Kept> {
    let b = std::fs::read(path).ok()?;
    let mut r = Reader { b: &b, at: 0 };
    if r.take(8)? != MAGIC || r.var()? != u64::from(VERSION) {
        return None;
    }
    let mut clusters = Vec::new();
    let mut last = 0u64;
    for _ in 0..r.var()? {
        let offset = last.checked_add(r.var()?)?;
        let data_start = offset.checked_add(r.var()?)?;
        let end = data_start.checked_add(r.var()?)?;
        clusters.push(ClusterInfo { offset, data_start, end, timestamp: r.var()? });
        last = offset;
    }
    let mut level1 = Vec::new();
    for _ in 0..r.var()? {
        level1.push((u32::try_from(r.var()?).ok()?, r.var()?));
    }
    let mut warnings = Vec::new();
    for _ in 0..r.var()? {
        let n = usize::try_from(r.var()?).ok()?;
        warnings.push(String::from_utf8(r.take(n)?.to_vec()).ok()?);
    }
    let mut tracks = Vec::new();
    for _ in 0..r.var()? {
        let number = r.var()?;
        let n = usize::try_from(r.var()?).ok()?;
        let mut samples = Vec::with_capacity(n.min(1 << 24));
        let (mut offset, mut pts, mut cluster) = (0i64, 0i64, 0i64);
        for _ in 0..n {
            offset = offset.checked_add(r.zig()?)?;
            pts = pts.checked_add(r.zig()?)?;
            cluster = cluster.checked_add(r.zig()?)?;
            let size = u32::try_from(r.var()?).ok()?;
            let duration = r.var()?;
            let block_back = r.var()?;
            let lace = u16::try_from(r.var()?).ok()?;
            let flags = r.take(1)?[0];
            let addition = if flags & 2 != 0 { Some((r.var()?, u32::try_from(r.var()?).ok()?)) } else { None };
            let offset_u = u64::try_from(offset).ok()?;
            samples.push(Sample {
                offset: offset_u,
                size,
                pts,
                duration,
                keyframe: flags & 1 != 0,
                cluster: u32::try_from(cluster).ok()?,
                block_offset: offset_u.checked_sub(block_back)?,
                lace,
                addition,
            });
        }
        tracks.push((number, samples));
    }
    (r.at == b.len()).then_some(Kept { clusters, level1, warnings, tracks })
}

/// Kept in `path` (written aside, then put in place: never half a file). Failures are ignored:
/// the next open indexes the file again.
pub(crate) fn save(path: &std::path::Path, k: &Kept) {
    let mut w = Vec::new();
    w.extend_from_slice(MAGIC);
    var(&mut w, u64::from(VERSION));
    var(&mut w, k.clusters.len() as u64);
    let mut last = 0u64;
    for c in &k.clusters {
        var(&mut w, c.offset.wrapping_sub(last));
        var(&mut w, c.data_start.wrapping_sub(c.offset));
        var(&mut w, c.end.wrapping_sub(c.data_start));
        var(&mut w, c.timestamp);
        last = c.offset;
    }
    var(&mut w, k.level1.len() as u64);
    for &(id, pos) in &k.level1 {
        var(&mut w, u64::from(id));
        var(&mut w, pos);
    }
    var(&mut w, k.warnings.len() as u64);
    for s in &k.warnings {
        var(&mut w, s.len() as u64);
        w.extend_from_slice(s.as_bytes());
    }
    var(&mut w, k.tracks.len() as u64);
    for (number, samples) in &k.tracks {
        var(&mut w, *number);
        var(&mut w, samples.len() as u64);
        let (mut offset, mut pts, mut cluster) = (0i64, 0i64, 0i64);
        for s in samples {
            zig(&mut w, (s.offset as i64).wrapping_sub(offset));
            zig(&mut w, s.pts.wrapping_sub(pts));
            zig(&mut w, i64::from(s.cluster).wrapping_sub(cluster));
            (offset, pts, cluster) = (s.offset as i64, s.pts, i64::from(s.cluster));
            var(&mut w, u64::from(s.size));
            var(&mut w, s.duration);
            var(&mut w, s.offset.saturating_sub(s.block_offset));
            var(&mut w, u64::from(s.lace));
            w.push(u8::from(s.keyframe) | if s.addition.is_some() { 2 } else { 0 });
            if let Some((o, n)) = s.addition {
                var(&mut w, o);
                var(&mut w, u64::from(n));
            }
        }
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = path.with_extension("mkvidx.tmp");
    if std::fs::write(&tmp, &w).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Forget a cache file that did not match its file.
pub(crate) fn drop_file(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

fn var(w: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        w.push((v as u8) | 0x80);
        v >>= 7;
    }
    w.push(v as u8);
}

fn zig(w: &mut Vec<u8>, v: i64) {
    var(w, ((v << 1) ^ (v >> 63)) as u64);
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }

    fn var(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self.b.get(self.at)?;
            self.at += 1;
            v |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }

    fn zig(&mut self) -> Option<i64> {
        let v = self.var()?;
        Some(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
}

/// FNV-1a, 64-bit.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ u64::from(x)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_index_reads_back_as_it_was() {
        let s = |offset, pts, keyframe, addition| Sample { offset, size: 1234, pts, duration: 40, keyframe, cluster: 3, block_offset: offset - 7, lace: 0, addition };
        let k = Kept {
            clusters: vec![ClusterInfo { offset: 100, data_start: 112, end: 9000, timestamp: 0 }, ClusterInfo { offset: 9000, data_start: 9012, end: 20000, timestamp: 5000 }],
            level1: vec![(0x1C53_BB6B, 20000)],
            warnings: vec!["damaged cluster".into()],
            tracks: vec![(1, vec![s(120, 0, true, None), s(4000, 40, false, Some((5300, 20)))]), (2, vec![s(200, -21, true, None)])],
        };
        let dir = std::env::temp_dir().join(format!("fc-mkvidx-{}", std::process::id()));
        let p = dir.join("x.mkvidx");
        save(&p, &k);
        let r = load(&p).expect("read back");
        assert_eq!(r.clusters, k.clusters);
        assert_eq!(r.level1, k.level1);
        assert_eq!(r.warnings, k.warnings);
        assert_eq!(r.tracks.len(), 2);
        for ((n0, a), (n1, b)) in r.tracks.iter().zip(&k.tracks) {
            assert_eq!(n0, n1);
            assert_eq!(a, b);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
