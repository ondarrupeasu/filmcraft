//! Clean-room MPEG-2 video encoder (ITU-T H.262 | ISO/IEC 13818-2) for XDCAM HD422-style
//! streams: 4:2:2 Profile at High Level, 1920x1080 interlaced at 25 frames per second (top field
//! first) as frame pictures with field or frame prediction and DCT per macroblock, long GOP
//! "IBBP" (N = 12, M = 3, open after the first), 50 Mbit/s constant bit rate.
//!
//! The rate control is TM5's (MPEG-2 Test Model 5): targets from the pictures' complexities,
//! activity-adapted quantiser scales; each picture is coded at the scale that meets its target
//! without emptying the VBV buffer (17 825 792 bits) of a decoder, and stuffed when the buffer
//! would overflow. Every GOP repeats the sequence header and its extensions (BT.709 colour).
//!
//! ```no_run
//! let (y, c) = (vec![16u8; 1920 * 1080], vec![128u8; 960 * 1080]);
//! let mut enc = filmcraft_mpeg2enc::Encoder::new();
//! let mut packets = enc.encode(&filmcraft_mpeg2enc::Frame { y: &y, cb: &c, cr: &c });
//! packets.extend(enc.flush());
//! ```

mod dct;
mod picture;
mod tables;

use filmcraft_bitstream::BitWriter;
use picture::{Coded, H, MB_H, PType, Pic, Setup, W, code_row, plan_row, reconstruct_row};

/// Picture size.
pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;
/// Bits a second and the VBV buffer.
pub const BIT_RATE: u64 = 50_000_000;
pub const VBV_BITS: u64 = 17_825_792;
const FPS: u64 = 25;
/// GOP length and anchor distance.
pub const GOP: u64 = 12;
const M: u64 = 3;

/// One frame of 8-bit 4:2:2 video, top field on the even lines: Y 1920 samples a line, Cb and
/// Cr 960, 1080 lines.
pub struct Frame<'a> {
    pub y: &'a [u8],
    pub cb: &'a [u8],
    pub cr: &'a [u8],
}

/// A coded picture, in coding order.
#[derive(Clone, Debug)]
pub struct Packet {
    pub data: Vec<u8>,
    /// Its frame number in display order.
    pub display: u64,
    /// An I picture, with the sequence and GOP headers before it.
    pub key: bool,
    /// A B picture (no picture is predicted from it).
    pub b: bool,
}

/// TM5's complexities and the GOP's remaining bits.
struct Rate {
    x: [f64; 3],
    remaining: f64,
    left: [u32; 3],
    /// VBV occupancy just before the next picture is taken out, in bits.
    fullness: f64,
}

pub struct Encoder {
    /// Frames waiting for the anchor after them: (display number, picture).
    pending: Vec<(u64, Pic)>,
    /// The last two anchors' reconstructions: older, newer.
    anchors: [Option<Pic>; 2],
    count: u64,
    /// Display number of the first frame of the current GOP.
    gop_start: u64,
    rate: Rate,
    first_gop: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    pub fn new() -> Self {
        let br = BIT_RATE as f64;
        Encoder {
            pending: Vec::new(),
            anchors: [None, None],
            count: 0,
            gop_start: 0,
            rate: Rate { x: [160.0 * br / 115.0, 60.0 * br / 115.0, 42.0 * br / 115.0], remaining: 0.0, left: [0; 3], fullness: VBV_BITS as f64 * 0.8 },
            first_gop: true,
        }
    }

    /// Take one frame; returns the pictures it lets be coded (an anchor, and the B pictures
    /// before it).
    pub fn encode(&mut self, f: &Frame) -> Vec<Packet> {
        let n = self.count;
        self.count += 1;
        let pic = padded(f);
        let anchor = n.is_multiple_of(M);
        if !anchor {
            self.pending.push((n, pic));
            return Vec::new();
        }
        self.code_anchor(n, pic)
    }

    /// Code what is left: the last frame becomes an anchor (P) for those before it.
    pub fn flush(&mut self) -> Vec<Packet> {
        let mut out = Vec::new();
        if let Some((n, pic)) = self.pending.pop() {
            out = self.code_anchor(n, pic);
        }
        if let Some(last) = out.last_mut() {
            last.data.extend_from_slice(&[0, 0, 1, 0xB7]); // sequence_end_code
        }
        out
    }

    fn code_anchor(&mut self, n: u64, pic: Pic) -> Vec<Packet> {
        let mut out = Vec::new();
        let intra = n.is_multiple_of(GOP);
        let ptype = if intra { PType::I } else { PType::P };
        if intra {
            // leading B pictures (open GOP) belong to this GOP in display
            self.gop_start = n - self.pending.len() as u64;
            self.start_gop();
        }
        let [r0, r1] = std::mem::take(&mut self.anchors);
        let newest = r1.or(r0);
        let newer = newest.as_ref();
        let (data, recon) = self.code_picture(ptype, n, &pic, if intra { None } else { newer }, None, intra);
        out.push(Packet { data, display: n, key: intra, b: false });
        let recon = recon.unwrap_or(pic);
        let older = newest;
        // the B pictures between the two anchors
        let pending = std::mem::take(&mut self.pending);
        for (k, bp) in pending {
            let (data, _) = self.code_picture(PType::B, k, &bp, older.as_ref(), Some(&recon), false);
            out.push(Packet { data, display: k, key: false, b: true });
        }
        self.anchors = [older, Some(recon)];
        out
    }

    fn start_gop(&mut self) {
        let r = &mut self.rate;
        r.remaining += (BIT_RATE / FPS * GOP) as f64;
        r.left = [1, (GOP / M - 1) as u32, (GOP - GOP / M) as u32];
    }

    /// Code one picture with its headers (and the sequence and GOP headers before an I
    /// picture); returns the bytes and, for anchors, the reconstruction.
    fn code_picture(&mut self, t: PType, n: u64, src: &Pic, fwd: Option<&Pic>, bwd: Option<&Pic>, gop: bool) -> (Vec<u8>, Option<Pic>) {
        let f_code = match t {
            PType::I => [15, 15],
            PType::P => [5, 15],
            PType::B => [4, 4],
        };
        let setup = Setup { ptype: t, src, fwd, bwd, f_code };
        let plans = rows(|r| plan_row(&setup, r));
        let acts: Vec<f64> = plans.iter().flatten().map(picture_act).collect();
        let mean_act = acts.iter().sum::<f64>() / acts.len().max(1) as f64;

        // TM5 target, within what the VBV buffer holds
        let k = match t {
            PType::I => 0,
            PType::P => 1,
            PType::B => 2,
        };
        let r = &self.rate;
        let (kp, kb) = (1.0, 1.4);
        let [xi, xp, xb] = r.x;
        let [_, np, nb] = r.left.map(f64::from);
        let floor = (BIT_RATE / (8 * FPS)) as f64;
        let target = match t {
            PType::I => r.remaining / (1.0 + np * xp / (xi * kp) + nb * xb / (xi * kb)),
            PType::P => r.remaining / (np + nb * kp * xb / (kb * xp)),
            PType::B => r.remaining / (nb + np * kb * xp / (kp * xb)),
        }
        .max(floor);
        let headers = if gop { 8 * (sequence_headers().len() + gop_header(n - self.gop_start, self.first_gop, self.gop_start).len()) } else { 0 };
        let room = r.fullness - headers as f64 - 8.0 * 64.0;
        let target = target.min(room * 0.95);

        // the quantiser scale: from the complexity, then corrected by trial
        let mut q = (r.x[k] / target.max(1.0)).clamp(1.0, 112.0);
        let mut coded = code_all(&setup, &plans, q, mean_act);
        for _ in 0..4 {
            let bits = coded.0.bits as f64;
            let over = bits > room;
            if !over && (bits - target).abs() < target * 0.06 {
                break;
            }
            let nq = (q * bits / target).clamp(1.0, 112.0);
            if (nq - q).abs() < 0.25 && !over {
                break;
            }
            q = if over { nq.max(q + 1.0) } else { nq };
            coded = code_all(&setup, &plans, q, mean_act);
        }
        while coded.0.bits as f64 > room && q < 112.0 {
            q = (q * 1.25).min(112.0);
            coded = code_all(&setup, &plans, q, mean_act);
        }
        let (c, row_codings) = coded;

        // the VBV buffer: the picture leaves, a frame period of bits arrives; stuff what would
        // overflow
        let mut bits = (c.bits + headers + 8 * 4 + 8 * picture_header_len(t)) as f64;
        let per_frame = (BIT_RATE / FPS) as f64;
        let rate = &mut self.rate;
        let vbv_delay = ((rate.fullness - headers as f64) * 90_000.0 / BIT_RATE as f64).round().clamp(0.0, 65_534.0) as u32;
        let overflow = rate.fullness - bits + per_frame - VBV_BITS as f64;
        let stuffing = if overflow > 0.0 { (overflow / 8.0).ceil() as usize } else { 0 };
        bits += 8.0 * stuffing as f64;
        rate.fullness = rate.fullness - bits + per_frame;
        rate.x[k] = bits * c.mean_q;
        rate.remaining -= bits;
        rate.left[k] = rate.left[k].saturating_sub(1);

        let mut out = Vec::with_capacity(c.bits / 8 + 1024 + stuffing);
        if gop {
            out.extend(sequence_headers());
            out.extend(gop_header(n - self.gop_start, self.first_gop, self.gop_start));
            self.first_gop = false;
        }
        out.extend(picture_header(t, (n - self.gop_start) as u32, vbv_delay, f_code));
        for s in &c.slices {
            out.extend_from_slice(s);
        }
        out.resize(out.len() + stuffing, 0);

        let recon = (t != PType::B).then(|| {
            let parts = rows(|r| reconstruct_row(&plans[r], &row_codings[r]));
            let mut p = Pic::blank();
            for (r, (y, c)) in parts.into_iter().enumerate() {
                p.y[r * 16 * W..(r + 1) * 16 * W].copy_from_slice(&y);
                for k in 0..2 {
                    p.c[k][r * 16 * picture::CW..(r + 1) * 16 * picture::CW].copy_from_slice(&c[k]);
                }
            }
            p
        });
        (out, recon)
    }
}

fn picture_act(m: &picture::MbPlan) -> f64 {
    m.activity()
}

/// Run `f` for every macroblock row (in parallel with the `threads` feature), in order.
fn rows<T: Send>(f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(feature = "threads")]
    {
        use rayon::prelude::*;
        (0..MB_H).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "threads"))]
    {
        (0..MB_H).map(f).collect()
    }
}

/// Every slice at base scale `q`.
fn code_all(s: &Setup, plans: &[Vec<picture::MbPlan>], q: f64, mean_act: f64) -> (Coded, Vec<picture::RowCoding>) {
    let parts = rows(|r| code_row(s, r, &plans[r], q, mean_act));
    let mut slices = Vec::with_capacity(MB_H);
    let mut codings = Vec::with_capacity(MB_H);
    let mut qs = 0.0;
    for (data, coding, mq) in parts {
        slices.push(data);
        codings.push(coding);
        qs += mq;
    }
    let bits = slices.iter().map(|s| s.len() * 8).sum();
    (Coded { slices, bits, mean_q: qs / MB_H as f64 }, codings)
}

/// The frame in coded size: 1088 lines, the last line repeated.
fn padded(f: &Frame) -> Pic {
    let mut p = Pic::blank();
    for line in 0..H {
        // keep the field structure: lines past 1080 repeat the last line of the same field
        let from = if line < HEIGHT { line } else { HEIGHT - 2 + line % 2 };
        p.y[line * W..(line + 1) * W].copy_from_slice(&f.y[from * W..(from + 1) * W]);
        let cw = W / 2;
        p.c[0][line * cw..(line + 1) * cw].copy_from_slice(&f.cb[from * cw..(from + 1) * cw]);
        p.c[1][line * cw..(line + 1) * cw].copy_from_slice(&f.cr[from * cw..(from + 1) * cw]);
    }
    p
}

/// sequence_header, sequence_extension and sequence_display_extension (§6.2.2).
fn sequence_headers() -> Vec<u8> {
    let mut w = BitWriter::new();
    let rate = (BIT_RATE / 400) as u32;
    let vbv = (VBV_BITS / 16_384) as u32;
    w.write_bits(0x0000_01B3, 32);
    w.write_bits(WIDTH as u32 & 0xFFF, 12);
    w.write_bits(HEIGHT as u32 & 0xFFF, 12);
    w.write_bits(3, 4); // aspect_ratio_information: 16:9
    w.write_bits(3, 4); // frame_rate_code: 25
    w.write_bits(rate & 0x3FFFF, 18);
    w.write_bit(true); // marker
    w.write_bits(vbv & 0x3FF, 10);
    w.write_bit(false); // constrained_parameters_flag
    w.write_bit(false); // load_intra_quantiser_matrix
    w.write_bit(false); // load_non_intra_quantiser_matrix
    w.write_bits(0x0000_01B5, 32);
    w.write_bits(1, 4); // sequence extension
    w.write_bits(0x82, 8); // profile_and_level_indication: 4:2:2 Profile at High Level
    w.write_bit(false); // progressive_sequence
    w.write_bits(2, 2); // chroma_format 4:2:2
    w.write_bits(0, 2);
    w.write_bits(0, 2);
    w.write_bits(rate >> 18, 12); // bit_rate_extension
    w.write_bit(true);
    w.write_bits(vbv >> 10, 8); // vbv_buffer_size_extension
    w.write_bit(false); // low_delay
    w.write_bits(0, 2);
    w.write_bits(0, 5);
    w.align_zero();
    w.write_bits(0x0000_01B5, 32);
    w.write_bits(2, 4); // sequence display extension
    w.write_bits(1, 3); // video_format: PAL
    w.write_bit(true); // colour_description
    w.write_bits(1, 8); // BT.709 primaries
    w.write_bits(1, 8); // BT.709 transfer
    w.write_bits(1, 8); // BT.709 matrix
    w.write_bits(WIDTH as u32, 14);
    w.write_bit(true);
    w.write_bits(HEIGHT as u32, 14);
    w.align_zero();
    w.finish()
}

/// group_of_pictures_header: the time code of the GOP's first frame in display order.
fn gop_header(_offset: u64, closed: bool, first_frame: u64) -> Vec<u8> {
    let mut w = BitWriter::new();
    let f = first_frame;
    let (h, m, s, p) = ((f / (FPS * 3600)) % 24, (f / (FPS * 60)) % 60, (f / FPS) % 60, f % FPS);
    w.write_bits(0x0000_01B8, 32);
    w.write_bit(false); // drop_frame_flag
    w.write_bits(h as u32, 5);
    w.write_bits(m as u32, 6);
    w.write_bit(true); // marker
    w.write_bits(s as u32, 6);
    w.write_bits(p as u32, 6);
    w.write_bit(closed); // closed_gop
    w.write_bit(false); // broken_link
    w.align_zero();
    w.finish()
}

/// Bytes of a picture header and its coding extension.
fn picture_header_len(t: PType) -> usize {
    picture_header(t, 0, 0, [15, 15]).len()
}

/// picture_header and picture_coding_extension (§6.2.3): a frame picture, top field first,
/// field or frame prediction and DCT, 10-bit DC, non-linear scale, intra VLC table one, the
/// alternate scan.
fn picture_header(t: PType, temporal_reference: u32, vbv_delay: u32, f_code: [u32; 2]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0x0000_0100, 32);
    w.write_bits(temporal_reference & 0x3FF, 10);
    w.write_bits(t.code(), 3);
    w.write_bits(vbv_delay & 0xFFFF, 16);
    if t != PType::I {
        w.write_bit(false); // full_pel_forward_vector
        w.write_bits(7, 3); // forward_f_code
    }
    if t == PType::B {
        w.write_bit(false);
        w.write_bits(7, 3);
    }
    w.write_bit(false); // extra_bit_picture
    w.align_zero();
    w.write_bits(0x0000_01B5, 32);
    w.write_bits(8, 4); // picture coding extension
    let (ff, bf) = match t {
        PType::I => (15, 15),
        PType::P => (f_code[0], 15),
        PType::B => (f_code[0], f_code[1]),
    };
    w.write_bits(ff, 4);
    w.write_bits(ff, 4);
    w.write_bits(bf, 4);
    w.write_bits(bf, 4);
    w.write_bits(picture::DC_PRECISION, 2);
    w.write_bits(3, 2); // picture_structure: frame
    w.write_bit(true); // top_field_first
    w.write_bit(false); // frame_pred_frame_dct
    w.write_bit(false); // concealment_motion_vectors
    w.write_bit(true); // q_scale_type: non-linear
    w.write_bit(true); // intra_vlc_format
    w.write_bit(true); // alternate_scan
    w.write_bit(false); // repeat_first_field
    w.write_bit(false); // chroma_420_type
    w.write_bit(false); // progressive_frame
    w.write_bit(false); // composite_display_flag
    w.align_zero();
    w.finish()
}

#[cfg(test)]
mod tests;
