//! Clean-room AVC-Intra Class 100 encoder for 1080i at 25 frames per second, written from
//! ITU-T H.264 (08/2021) and the public description of SMPTE RP 2027 Class 100 1080/50i in AMWA
//! AS-11 X2: H.264 High 4:2:2 Intra profile at level 4.1, 10-bit 4:2:2, CAVLC, MBAFF frames
//! (top field first) whose macroblock pairs are all field pairs, every picture IDR, and every
//! coded frame exactly [`FRAME_BYTES`] long (access unit delimiter, the parameter sets at the
//! start of every frame, the slices, and filler data).
//!
//! The coding is intra only and the slices say not to deblock, so the encoder's reconstruction
//! is what every decoder shows. The frame is cut in [`SLICES`] slices coded in parallel; the
//! frame's QP is the lowest that fits, then slices take one step finer while the frame still
//! fits.
//!
//! ```no_run
//! let (y, c) = (vec![512u16; 1920 * 1080], vec![512u16; 960 * 1080]);
//! let mut enc = filmcraft_avcintra::Encoder::new();
//! let frame = enc.encode(&filmcraft_avcintra::Frame { y: &y, cb: &c, cr: &c });
//! assert_eq!(frame.len(), filmcraft_avcintra::FRAME_BYTES);
//! ```

mod cavlc;
mod headers;
mod pred;
mod slice;
mod tables;
mod transform;

use headers::{MB_H, aud, filler, pps, sps};
use slice::{Planes, encode_slice};

pub use headers::{MB_W, nal};

/// Width and height of the pictures.
pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;
/// The size of every coded frame of Class 100 1080/50i (AS-11 X2: "the coded frame size will be
/// 568832 bytes, including the coded frame, sequence parameter set, picture parameter set, SEI
/// messages and/or padding").
pub const FRAME_BYTES: usize = 568_832;
/// Slices per frame (each a band of macroblock pair rows of both fields).
pub const SLICES: usize = 10;

/// One 1920x1080 frame of 10-bit 4:2:2 samples (BT.709, video range), top field on the even
/// lines: Y 1920 samples a line, Cb and Cr 960.
pub struct Frame<'a> {
    pub y: &'a [u16],
    pub cb: &'a [u16],
    pub cr: &'a [u16],
}

/// The encoder: it remembers the QP for the next frame.
pub struct Encoder {
    qp: i32,
    frames: u32,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// QP range used (QP_Y; 10-bit allows −12..51).
const QP_MIN: i32 = -8;
const QP_MAX: i32 = 51;

impl Encoder {
    pub fn new() -> Self {
        Encoder { qp: 16, frames: 0 }
    }

    /// Code one frame into exactly [`FRAME_BYTES`] bytes of Annex B byte stream: access unit
    /// delimiter, SPS, PPS, the slices, filler data.
    pub fn encode(&mut self, f: &Frame) -> Vec<u8> {
        let (sps, pps, aud) = (sps(), pps(), aud());
        // the filler NAL unit is at least 6 bytes
        let budget = FRAME_BYTES - (aud.len() + sps.len() + pps.len() + 6);
        let [t, b] = [0, 1].map(|parity| field(f, parity));
        let top = Planes { y: &t.0, cb: &t.1, cr: &t.2, width: WIDTH };
        let bottom = Planes { y: &b.0, cb: &b.1, cr: &b.2, width: WIDTH };
        // consecutive IDR pictures need different idr_pic_id
        let id = self.frames % 2;
        self.frames = self.frames.wrapping_add(1);
        let (qp, slices) = fit([&top, &bottom], self.qp, budget, id);
        self.qp = qp;
        let mut out = Vec::with_capacity(FRAME_BYTES);
        out.extend_from_slice(&aud);
        out.extend_from_slice(&sps);
        out.extend_from_slice(&pps);
        for s in &slices {
            out.extend_from_slice(s);
        }
        let rest = FRAME_BYTES - out.len();
        out.extend_from_slice(&filler(rest));
        debug_assert_eq!(out.len(), FRAME_BYTES);
        out
    }
}

/// One field of the frame (`parity` 0 top, 1 bottom) as planes of 34 macroblock rows (the
/// last line repeated below its 540).
fn field(f: &Frame, parity: usize) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let rows = MB_H * 16;
    let pick = |p: &[u16], w: usize| {
        let mut out = Vec::with_capacity(w * rows);
        for r in 0..rows {
            let line = (2 * r.min(HEIGHT / 2 - 1) + parity) * w;
            out.extend_from_slice(&p[line..line + w]);
        }
        out
    };
    (pick(f.y, WIDTH), pick(f.cb, WIDTH / 2), pick(f.cr, WIDTH / 2))
}

/// Macroblock pair rows of slice `s`.
fn bounds(s: usize) -> (usize, usize) {
    (s * MB_H / SLICES, (s + 1) * MB_H / SLICES)
}

/// Every slice at its QP.
fn code_slices(f: [&Planes; 2], qps: &[i32], id: u32) -> Vec<Vec<u8>> {
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        qps.par_iter().enumerate().map(|(s, &q)| encode_slice(f, bounds(s).0, bounds(s).1, q, id)).collect()
    }
    #[cfg(not(feature = "threads"))]
    {
        qps.iter().enumerate().map(|(s, &q)| encode_slice(f, bounds(s).0, bounds(s).1, q, id)).collect()
    }
}

fn total(slices: &[Vec<u8>]) -> usize {
    slices.iter().map(Vec::len).sum()
}

/// The lowest QP (searched from `start`) at which the frame fits in `budget` bytes, then the
/// slices that still fit one step finer; returns the frame's QP and its slices.
fn fit(f: [&Planes; 2], start: i32, budget: usize, id: u32) -> (i32, Vec<Vec<u8>>) {
    let mut qp = start.clamp(QP_MIN, QP_MAX);
    let mut slices = code_slices(f, &[qp; SLICES], id);
    // too big: coarser until it fits
    while total(&slices) > budget && qp < QP_MAX {
        qp = (qp + if total(&slices) > budget * 5 / 4 { 2 } else { 1 }).min(QP_MAX);
        slices = code_slices(f, &[qp; SLICES], id);
    }
    // room left: the whole frame finer while it fits
    while qp > QP_MIN && total(&slices) * 100 < budget * 90 {
        let finer = code_slices(f, &[qp - 1; SLICES], id);
        if total(&finer) > budget {
            break;
        }
        qp -= 1;
        slices = finer;
    }
    // then slice by slice, one step finer where it still fits
    if qp > QP_MIN {
        let finer = code_slices(f, &[qp - 1; SLICES], id);
        for (s, fs) in finer.into_iter().enumerate() {
            if total(&slices) - slices[s].len() + fs.len() <= budget {
                slices[s] = fs;
            }
        }
    }
    (qp, slices)
}

#[cfg(test)]
mod tests;
