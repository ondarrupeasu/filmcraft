//! A clean-room AC-3 (Dolby Digital) encoder, written from ATSC A/52:2012 beside the decoder it
//! is checked against: mono, stereo and 5.1 at 32 / 44.1 / 48 kHz, 32 to 640 kb/s.
//!
//! - Long blocks only (the 512-sample transform of §7.9.4 with its window), its analysis the
//!   transpose of the decoder's synthesis, scaled so they reconstruct.
//! - Exponents shared across blocks while the spectrum holds (a set when it changes), D15, D25
//!   or D45 by the bit rate, kept within the ±2 steps their differential coding allows.
//! - The parametric bit allocation is the decoder's own (§7.2), with the usual parameters; the
//!   SNR offset is the highest whose mantissas fit the frame. Dither for zero-bit mantissas.
//! - No coupling, no rematrixing, no dynamic range words; both CRCs (§7.10.1).

use crate::{AllocParams, BITRATES, Error, FASTGAIN, Globals, QNTZTAB, Result, bit_allocation, imdct};
use std::sync::OnceLock;

/// What to encode: the rate, the channels (1, 2, or 6 in WAV order: L R C LFE Ls Rs) and the
/// bit rate (kb/s: the nearest of Table 5.18 at or below it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub sample_rate: u32,
    pub channels: usize,
    pub bitrate_kbps: u32,
}

/// The analysis basis: the decoder's 512-sample synthesis of each coefficient, and the scale
/// that makes analysis then synthesis give the input back.
struct Basis {
    s: Vec<[f32; 512]>,
    scale: f32,
}

fn basis() -> &'static Basis {
    static B: OnceLock<Basis> = OnceLock::new();
    B.get_or_init(|| {
        let mut s = vec![[0f32; 512]; 256];
        for (k, out) in s.iter_mut().enumerate() {
            let mut x = [0f32; 256];
            x[k] = 1.0;
            imdct(&x, false, out);
        }
        let mut b = Basis { s, scale: 1.0 };
        // A signal through analysis and the decoder's overlap-add: the gain to take out.
        let x: Vec<f32> = (0..2048).map(|n| ((n as f32 * 0.071).sin() + 0.5 * (n as f32 * 0.37).cos()) * 0.3).collect();
        let blocks: Vec<[f32; 256]> = (0..7).map(|k| analyze(&b, &x[256 * k..256 * k + 512])).collect();
        let (mut num, mut den) = (0f64, 0f64);
        let mut prev = [0f32; 512];
        for (k, c) in blocks.iter().enumerate() {
            let mut buf = [0f32; 512];
            imdct(c, false, &mut buf);
            if k > 0 {
                for n in 0..256 {
                    let y = 2.0 * (buf[n] + prev[256 + n]);
                    let want = x[256 * k + n];
                    num += f64::from(y * want);
                    den += f64::from(want * want);
                }
            }
            prev = buf;
        }
        b.scale = (den / num) as f32;
        b
    })
}

fn analyze(b: &Basis, x: &[f32]) -> [f32; 256] {
    let mut out = [0f32; 256];
    for (k, o) in out.iter_mut().enumerate() {
        let s = &b.s[k];
        let mut acc = 0f32;
        for n in 0..512 {
            acc += s[n] * x[n];
        }
        *o = acc * b.scale;
    }
    out
}

/// The exponent of a coefficient: the left shift that brings it into [0.5, 1) (0 … 24).
fn exponent(c: f32) -> u8 {
    let a = c.abs();
    if a >= 0.5 || !a.is_finite() {
        return 0;
    }
    if a == 0.0 {
        return 24;
    }
    ((-a.log2()).floor() as i32).clamp(0, 24) as u8
}

/// CRC-16 (x¹⁶ + x¹⁵ + x² + 1), zero start, most significant bit first.
fn crc16(data: &[u8]) -> u16 {
    let mut c: u16 = 0;
    for &b in data {
        c ^= u16::from(b) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
        }
    }
    c
}

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
    bits: usize,
}

impl BitWriter {
    fn new(bytes: usize) -> Self {
        Self { buf: Vec::with_capacity(bytes), acc: 0, n: 0, bits: 0 }
    }
    fn put(&mut self, v: u32, w: u32) {
        if w == 0 {
            return;
        }
        self.acc = (self.acc << w) | u64::from(v & ((1u64 << w) - 1) as u32);
        self.n += w;
        self.bits += w as usize;
        while self.n >= 8 {
            self.buf.push((self.acc >> (self.n - 8)) as u8);
            self.n -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push((self.acc << (8 - self.n)) as u8);
        }
        self.buf
    }
}

/// One channel's exponents for a frame: which blocks send a new set (and how), and the
/// exponents every block decodes to.
struct Exps {
    /// Per block: 0 reuse, else the strategy (1 D15, 2 D25, 3 D45; the LFE channel: 1).
    strategy: [u8; 6],
    exps: [[u8; 256]; 6],
    /// Per block with a new set: the absolute exponent and the 7-bit groups.
    coded: [(u8, Vec<u8>); 6],
}

/// Exponents `raw` (to `end`) as strategy `st` codes them: the levels, kept within ±2 of their
/// neighbours and never above what a bin needs, the decoded exponents and their groups.
fn code_exponents(raw: &[u8; 256], end: usize, st: u8, lfe: bool) -> ([u8; 256], u8, Vec<u8>) {
    let gs = if lfe { 1 } else { 1usize << (st - 1) };
    let ngrps = if lfe { 2 } else { (end - 1 + 3 * gs - 3) / (3 * gs) };
    let mut lv = vec![0i32; 1 + 3 * ngrps];
    lv[0] = i32::from(raw[0]).min(15);
    for j in 1..lv.len() {
        let (a, b) = (1 + (j - 1) * gs, (1 + j * gs).min(end));
        lv[j] = if a < b { raw[a..b].iter().map(|&e| i32::from(e)).min().unwrap_or(24) } else { lv[j - 1] };
    }
    for j in 1..lv.len() {
        lv[j] = lv[j].min(lv[j - 1] + 2);
    }
    for j in (0..lv.len() - 1).rev() {
        lv[j] = lv[j].min(lv[j + 1] + 2);
    }
    let mut exps = [0u8; 256];
    exps[0] = lv[0] as u8;
    for j in 1..lv.len() {
        for bin in 1 + (j - 1) * gs..(1 + j * gs).min(256) {
            exps[bin] = lv[j] as u8;
        }
    }
    let groups = (0..ngrps)
        .map(|g| {
            let d = |j: usize| (lv[j] - lv[j - 1] + 2) as u8;
            d(3 * g + 1) * 25 + d(3 * g + 2) * 5 + d(3 * g + 3)
        })
        .collect();
    (exps, lv[0] as u8, groups)
}

/// The exponent sets of a channel's six blocks: a new set where the spectrum moves away from
/// the last one (`always_new`: every block), each set the smallest exponent of its blocks.
fn plan_exponents(raw: &[[u8; 256]; 6], end: usize, st: u8, lfe: bool, share: bool) -> Exps {
    let mut starts = vec![0usize];
    if share {
        for b in 1..6 {
            let last = *starts.last().unwrap_or(&0);
            let moved = (0..end).map(|i| (i32::from(raw[b][i]) - i32::from(raw[last][i])).abs()).sum::<i32>() as f32 / end as f32;
            if moved > 1.5 {
                starts.push(b);
            }
        }
    } else {
        starts = (0..6).collect();
    }
    let mut e = Exps { strategy: [0; 6], exps: [[0; 256]; 6], coded: Default::default() };
    for (k, &s) in starts.iter().enumerate() {
        let stop = starts.get(k + 1).copied().unwrap_or(6);
        let mut m = [24u8; 256];
        for blk in raw.iter().take(stop).skip(s) {
            for i in 0..end {
                m[i] = m[i].min(blk[i]);
            }
        }
        let (exps, abs, groups) = code_exponents(&m, end, st, lfe);
        e.strategy[s] = if lfe { 1 } else { st };
        e.coded[s] = (abs, groups);
        for b in s..stop {
            e.exps[b] = exps;
        }
    }
    e
}

/// A mantissa's code for allocation `bap` (and its bits, 0 for the grouped ones).
fn quantize(m: f32, bap: u8) -> (u32, u32) {
    let sym = |levels: i32| (((m * levels as f32 + (levels - 1) as f32) / 2.0).round() as i32).clamp(0, levels - 1) as u32;
    match bap {
        1 => (sym(3), 0),
        2 => (sym(5), 0),
        3 => (sym(7), 3),
        4 => (sym(11), 0),
        5 => (sym(15), 4),
        _ => {
            let bits = QNTZTAB[bap as usize & 15];
            let half = 1i32 << (bits - 1);
            let v = ((m * half as f32).round() as i32).clamp(-half, half - 1);
            ((v as u32) & ((1u32 << bits) - 1), bits)
        }
    }
}

pub struct Encoder {
    acmod: u8,
    lfe: bool,
    /// The input channel of each coded full-bandwidth channel (coded order), and the LFE's.
    map: Vec<usize>,
    lfe_in: Option<usize>,
    fscod: u8,
    frmsizecod: u8,
    frame_bytes: usize,
    chbwcod: u8,
    endmant: usize,
    strategy: u8,
    /// The input, from the start of the next frame's first window (256 samples before it).
    buf: Vec<Vec<f32>>,
    taken: u64,
    frames: u64,
    /// crc1's value by the CRC it causes over the first 5/8 (linear: solved once).
    crc1_inverse: Vec<u16>,
}

impl Encoder {
    pub fn new(cfg: Config) -> Result<Self> {
        let fscod = match cfg.sample_rate {
            48_000 => 0u8,
            44_100 => 1,
            32_000 => 2,
            r => return Err(Error::Unsupported(format!("AC-3 at {r} Hz (48, 44.1 or 32 kHz)"))),
        };
        let (acmod, lfe, map, lfe_in) = match cfg.channels {
            1 => (1u8, false, vec![0], None),
            2 => (2, false, vec![0, 1], None),
            6 => (7, true, vec![0, 2, 1, 4, 5], Some(3)),
            n => return Err(Error::Unsupported(format!("AC-3 of {n} channels (1, 2 or 5.1)"))),
        };
        let k = BITRATES.iter().rposition(|&b| b <= cfg.bitrate_kbps.max(32)).unwrap_or(0);
        let kbps = BITRATES[k];
        let words = match fscod {
            0 => kbps as usize * 2,
            2 => kbps as usize * 3,
            _ => (kbps as usize * 1536 * 1000) / 44_100 / 16,
        };
        let frame_bytes = words * 2;
        let per = kbps as usize / map.len();
        // The bandwidth (end bin) and how exponents are coded, by the rate each channel has.
        let chbwcod: u8 = match per {
            p if p >= 64 => 46,
            p if p >= 48 => 40,
            p if p >= 32 => 30,
            _ => 20,
        };
        let strategy = match per {
            p if p >= 160 => 1,
            p if p >= 80 => 2,
            _ => 3,
        };
        let endmant = (usize::from(chbwcod) + 12) * 3 + 37;
        // crc1: the CRC over the first 5/8 is that of the rest XOR this linear map of crc1.
        let f58 = 2 * ((words >> 1) + (words >> 3));
        let mut cols = [0u16; 16];
        for (i, c) in cols.iter_mut().enumerate() {
            let mut b = vec![0u8; f58 - 2];
            let v = 1u16 << i;
            b[0] = (v >> 8) as u8;
            b[1] = v as u8;
            *c = crc16(&b);
        }
        let mut crc1_inverse = vec![0u16; 65_536];
        for v in 0..=u16::MAX {
            let mut a = 0u16;
            for (i, c) in cols.iter().enumerate() {
                if v & (1 << i) != 0 {
                    a ^= c;
                }
            }
            crc1_inverse[a as usize] = v;
        }
        let _ = basis();
        Ok(Self {
            acmod,
            lfe,
            map,
            lfe_in,
            fscod,
            frmsizecod: (k * 2) as u8,
            frame_bytes,
            chbwcod,
            endmant,
            strategy,
            buf: vec![vec![0f32; 256]; cfg.channels],
            taken: 0,
            frames: 0,
            crc1_inverse,
        })
    }

    /// Bytes in each syncframe.
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// The bit rate used (kb/s).
    pub fn bitrate_kbps(&self) -> u32 {
        BITRATES[usize::from(self.frmsizecod / 2)]
    }

    /// The decoder's output is this many samples later than the input.
    pub fn delay(&self) -> u32 {
        256
    }

    /// The `dac3` box of an MPEG-4 track (ETSI TS 102 366 Annex F).
    pub fn dac3(&self) -> Vec<u8> {
        let v = (u32::from(self.fscod) << 22) | (8 << 17) | (u32::from(self.acmod) << 11) | (u32::from(self.lfe) << 10) | (u32::from(self.frmsizecod / 2) << 5);
        v.to_be_bytes()[1..].to_vec()
    }

    /// Samples in (planar, as the configuration's channels): the syncframes complete.
    pub fn encode(&mut self, planar: &[Vec<f32>]) -> Vec<Vec<u8>> {
        let n = planar.iter().map(Vec::len).min().unwrap_or(0);
        for (ch, b) in self.buf.iter_mut().enumerate() {
            b.extend_from_slice(planar.get(ch).map_or(&[][..], |p| &p[..n]));
            if planar.get(ch).is_none() {
                b.extend(std::iter::repeat_n(0.0, n));
            }
        }
        self.taken += n as u64;
        let mut out = Vec::new();
        while self.buf[0].len() >= 1536 + 256 {
            out.push(self.frame());
        }
        out
    }

    /// The last syncframes: until the decoder has given back every sample put in.
    pub fn flush(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        let need = (self.taken + 256).div_ceil(1536);
        while self.frames < need {
            for b in &mut self.buf {
                b.resize(b.len().max(1536 + 256), 0.0);
            }
            out.push(self.frame());
        }
        out
    }

    /// One syncframe from the first 1536 + 256 samples of the input (1536 then go).
    fn frame(&mut self) -> Vec<u8> {
        let b = basis();
        let nf = self.map.len();
        // Coefficients and their exponents: coded channels, then the LFE.
        let chans: Vec<usize> = self.map.iter().copied().chain(self.lfe_in).collect();
        let mut coefs = vec![[[0f32; 256]; 6]; chans.len()];
        let mut raw = vec![[[24u8; 256]; 6]; chans.len()];
        for (c, &src) in chans.iter().enumerate() {
            let end = if c < nf { self.endmant } else { 7 };
            for blk in 0..6 {
                coefs[c][blk] = analyze(b, &self.buf[src][256 * blk..256 * blk + 512]);
                for i in 0..end {
                    raw[c][blk][i] = exponent(coefs[c][blk][i]);
                }
            }
        }
        // Exponent plans, cheaper ones when the frame cannot hold the first.
        let attempts: [(u8, bool); 3] = [(self.strategy, true), (self.strategy, false), (3, false)];
        let mut best = None;
        for (k, &(st, share)) in attempts.iter().enumerate() {
            // (share: a set when the spectrum moves; else one set for the whole frame)
            let plans: Vec<Exps> = (0..chans.len())
                .map(|c| {
                    let lfe = c >= nf;
                    let end = if lfe { 7 } else { self.endmant };
                    if share { plan_exponents(&raw[c], end, st, lfe, true) } else { plan_one(&raw[c], end, st, lfe) }
                })
                .collect();
            let (mut lo, mut hi) = (0i32, 1023i32);
            if self.write(&plans, &coefs, 0).is_none() {
                if k + 1 < attempts.len() {
                    continue;
                }
                best = Some(self.write(&plans, &coefs, 0).unwrap_or_else(|| vec![0; self.frame_bytes]));
                break;
            }
            while lo < hi {
                let mid = (lo + hi + 1) / 2;
                if self.write(&plans, &coefs, mid).is_some() { lo = mid } else { hi = mid - 1 }
            }
            best = self.write(&plans, &coefs, lo);
            break;
        }
        for buf in &mut self.buf {
            buf.drain(..1536);
        }
        self.frames += 1;
        best.unwrap_or_else(|| vec![0; self.frame_bytes])
    }

    /// The syncframe with SNR offset `snr` (csnroffst × 16 + fsnroffst), or none when it does
    /// not fit.
    fn write(&self, plans: &[Exps], coefs: &[[[f32; 256]; 6]], snr: i32) -> Option<Vec<u8>> {
        let nf = self.map.len();
        let (csnr, fsnr) = (snr >> 4, snr & 15);
        let g = Globals { fscod: usize::from(self.fscod), sdcycod: 2, fdcycod: 1, sgaincod: 1, dbpbcod: 2, floorcod: 7 };
        let offset = (((csnr - 15) << 4) + fsnr) << 2;
        let budget = self.frame_bytes * 8 - 18;
        let mut w = BitWriter::new(self.frame_bytes);
        // syncinfo
        w.put(0x0B77, 16);
        w.put(0, 16);
        w.put(u32::from(self.fscod), 2);
        w.put(u32::from(self.frmsizecod), 6);
        // bsi
        w.put(8, 5);
        w.put(0, 3);
        w.put(u32::from(self.acmod), 3);
        if self.acmod & 1 != 0 && self.acmod != 1 {
            w.put(0, 2); // cmixlev −3 dB
        }
        if self.acmod & 4 != 0 {
            w.put(0, 2); // surmixlev −3 dB
        }
        if self.acmod == 2 {
            w.put(0, 2); // dsurmod
        }
        w.put(u32::from(self.lfe), 1);
        w.put(31, 5); // dialnorm: −31 dB, played as it is
        w.put(0, 3); // compre, langcode, audprodie
        w.put(0, 1); // copyrightb
        w.put(1, 1); // origbs
        w.put(0, 3); // timecod1e, timecod2e, addbsie
        for blk in 0..6 {
            for _ in 0..nf {
                w.put(0, 1); // blksw
            }
            for _ in 0..nf {
                w.put(1, 1); // dithflag
            }
            w.put(0, 1); // dynrnge
            if blk == 0 {
                w.put(1, 1); // cplstre
                w.put(0, 1); // cplinu
            } else {
                w.put(0, 1);
            }
            if self.acmod == 2 {
                if blk == 0 {
                    w.put(1, 1); // rematstr
                    w.put(0, 4); // rematflg (four bands)
                } else {
                    w.put(0, 1);
                }
            }
            for p in &plans[..nf] {
                w.put(u32::from(p.strategy[blk]), 2);
            }
            if self.lfe {
                w.put(u32::from(plans[nf].strategy[blk]), 1);
            }
            for p in &plans[..nf] {
                if p.strategy[blk] != 0 {
                    w.put(u32::from(self.chbwcod), 6);
                }
            }
            for p in &plans[..nf] {
                if p.strategy[blk] != 0 {
                    w.put(u32::from(p.coded[blk].0), 4);
                    for &gr in &p.coded[blk].1 {
                        w.put(u32::from(gr), 7);
                    }
                    w.put(0, 2); // gainrng
                }
            }
            if self.lfe && plans[nf].strategy[blk] != 0 {
                w.put(u32::from(plans[nf].coded[blk].0), 4);
                for &gr in &plans[nf].coded[blk].1 {
                    w.put(u32::from(gr), 7);
                }
            }
            if blk == 0 {
                w.put(1, 1); // baie
                w.put(2, 2);
                w.put(1, 2);
                w.put(1, 2);
                w.put(2, 2);
                w.put(7, 3);
                w.put(1, 1); // snroffste
                w.put(csnr as u32, 6);
                for _ in 0..nf + usize::from(self.lfe) {
                    w.put(fsnr as u32, 4);
                    w.put(4, 3); // fgaincod
                }
            } else {
                w.put(0, 1);
                w.put(0, 1);
            }
            w.put(0, 1); // deltbaie
            w.put(0, 1); // skiple
            // Mantissas, coded channels then the LFE; the grouped ones three (or two) to a code,
            // written where the first of them is.
            let mut codes: Vec<(u8, u32, u32)> = Vec::new();
            for (c, p) in plans.iter().enumerate() {
                let end = if c < nf { self.endmant } else { 7 };
                let mut bap = [0u8; 256];
                if snr != 0 {
                    let ap = AllocParams { start: 0, end, fgain: FASTGAIN[4], snroffset: offset, leak: None, dba: None };
                    bit_allocation(&p.exps[blk], &ap, &g, &mut bap);
                }
                for i in 0..end {
                    if bap[i] == 0 {
                        continue;
                    }
                    let m = coefs[c][blk][i] * 2f32.powi(i32::from(p.exps[blk][i]));
                    let (q, bits) = quantize(m.clamp(-0.999_999, 0.999_999), bap[i]);
                    codes.push((bap[i], q, bits));
                }
            }
            let mut pend: [Vec<u32>; 3] = Default::default();
            let group = |bap: u8| match bap {
                1 => Some((0usize, 3usize, 3u32, 5u32)),
                2 => Some((1, 3, 5, 7)),
                4 => Some((2, 2, 11, 7)),
                _ => None,
            };
            // First pass: every grouped code in its group, in order.
            for &(bap, q, _) in &codes {
                if let Some((k, ..)) = group(bap) {
                    pend[k].push(q);
                }
            }
            let mut seen = [0usize; 3];
            for &(bap, q, bits) in &codes {
                match group(bap) {
                    Some((k, size, levels, width)) => {
                        if seen[k] % size == 0 {
                            let start = seen[k];
                            let v = (0..size).fold(0u32, |acc, j| acc * levels + pend[k].get(start + j).copied().unwrap_or(0));
                            w.put(v, width);
                        }
                        seen[k] += 1;
                    }
                    None => w.put(q, bits),
                }
            }
            if w.bits > budget {
                return None;
            }
        }
        // auxiliary data (none: zeros to the end), auxdatae, crcrsv, crc2
        while w.bits < budget {
            let n = (budget - w.bits).min(32) as u32;
            w.put(0, n);
        }
        w.put(0, 1);
        w.put(0, 1);
        w.put(0, 16);
        let mut f = w.finish();
        f.resize(self.frame_bytes, 0);
        let words = self.frame_bytes / 2;
        let f58 = 2 * ((words >> 1) + (words >> 3));
        let rest = crc16(&f[2..f58]);
        let c1 = self.crc1_inverse[rest as usize];
        f[2] = (c1 >> 8) as u8;
        f[3] = c1 as u8;
        let n = f.len();
        let c2 = crc16(&f[f58..n - 2]);
        f[n - 2] = (c2 >> 8) as u8;
        f[n - 1] = c2 as u8;
        Some(f)
    }
}

/// One exponent set for all six blocks (the smallest of each bin).
fn plan_one(raw: &[[u8; 256]; 6], end: usize, st: u8, lfe: bool) -> Exps {
    let mut m = [24u8; 256];
    for blk in raw {
        for i in 0..end {
            m[i] = m[i].min(blk[i]);
        }
    }
    let (exps, abs, groups) = code_exponents(&m, end, st, lfe);
    let mut e = Exps { strategy: [0; 6], exps: [exps; 6], coded: Default::default() };
    e.strategy[0] = if lfe { 1 } else { st };
    e.coded[0] = (abs, groups);
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Decoder;

    fn tone(n: usize, f: f32, a: f32) -> Vec<f32> {
        (0..n).map(|i| a * (2.0 * std::f32::consts::PI * f * i as f32 / 48_000.0).sin()).collect()
    }

    /// Encoded, then decoded by the crate's decoder: the same sound 256 samples later, each
    /// channel where it was; both CRCs right; the frames as long as the rate says.
    #[test]
    fn what_it_encodes_decodes_back() {
        for (chans, kbps) in [(1usize, 96u32), (2, 192), (6, 448)] {
            let n = 48_000;
            let input: Vec<Vec<f32>> = (0..chans).map(|c| if chans == 6 && c == 3 { tone(n, 60.0, 0.5) } else { tone(n, 300.0 + 220.0 * c as f32, 0.5) }).collect();
            let mut e = Encoder::new(Config { sample_rate: 48_000, channels: chans, bitrate_kbps: kbps }).expect("encoder");
            let mut frames = Vec::new();
            for k in (0..n).step_by(1000) {
                let part: Vec<Vec<f32>> = input.iter().map(|c| c[k..(k + 1000).min(n)].to_vec()).collect();
                frames.extend(e.encode(&part));
            }
            frames.extend(e.flush());
            assert_eq!(frames.len(), (n + 256).div_ceil(1536), "{chans} channels");
            let mut d = Decoder::new();
            let mut out: Vec<Vec<f32>> = vec![Vec::new(); chans];
            for f in &frames {
                assert_eq!(f.len(), e.frame_bytes());
                let h = crate::parse_header(f).expect("header");
                assert_eq!(h.frame_bytes, f.len());
                assert_eq!(h.bitrate_kbps, kbps);
                let words = f.len() / 2;
                let f58 = 2 * ((words >> 1) + (words >> 3));
                assert_eq!(crc16(&f[2..f58]), 0, "crc1");
                assert_eq!(crc16(&f[f58..]), 0, "crc2");
                let got = d.decode(f).expect("decodes");
                for (c, o) in out.iter_mut().enumerate() {
                    o.extend_from_slice(&got.channels[c]);
                }
            }
            for c in 0..chans {
                let (x, y) = (&input[c][2000..40_000], &out[c][2256..40_256]);
                let err: f32 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / x.len() as f32;
                let sig: f32 = x.iter().map(|a| a * a).sum::<f32>() / x.len() as f32;
                let snr = 10.0 * (sig / err.max(1e-12)).log10();
                assert!(snr > 25.0, "{chans} ch, channel {c}: {snr:.1} dB");
            }
        }
    }

    #[test]
    fn a_full_scale_sound_does_not_clip() {
        let n = 9_000;
        let x = tone(n, 1000.0, 0.999);
        let mut e = Encoder::new(Config { sample_rate: 48_000, channels: 1, bitrate_kbps: 192 }).unwrap();
        let mut frames = e.encode(std::slice::from_ref(&x));
        frames.extend(e.flush());
        let mut d = Decoder::new();
        let y: Vec<f32> = frames.iter().flat_map(|f| d.decode(f).unwrap().channels[0].clone()).collect();
        let peak = y[2000..8000].iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!((peak - 0.999).abs() < 0.05, "peak {peak}");
    }
}
