//! Coding one slice of a field picture: Intra_4x4 / Intra_16x16 luma and 4:2:2 chroma
//! macroblocks, mode decision by SATD, reconstruction for the following blocks' prediction, and
//! the CAVLC macroblock layer (§7.3.5).

use filmcraft_bitstream::BitWriter;

use crate::cavlc::{NC_CDC422, nc_of, write_residual};
use crate::headers::{CHROMA_QP_OFFSET, MB_W, NAL_IDR, nal, slice_header};
use crate::pred::{
    Edge4, EdgeChroma, EdgeLuma, MAX, mode_chroma_available, mode4_available, mode16_available, pred_chroma422, pred4x4, pred16x16,
};
use crate::tables::{BLK4_XY, CBP_INTRA_TO_CODE, CDC422_POS, CHROMA_4X4, DEFAULT_4X4_INTRA, FIELD4, qpc, weights};
use crate::transform::{Scale, dequant_chroma_dc422, dequant_luma_dc, dequant4, fdct4, idct4, quant_chroma_dc422, quant_luma_dc, quant4, satd4};

/// One field (or any picture) to code: 10-bit planes, chroma half width, all 16·rows tall.
pub struct Planes<'a> {
    pub y: &'a [u16],
    pub cb: &'a [u16],
    pub cr: &'a [u16],
    /// Luma width (the chroma is half of it).
    pub width: usize,
}

/// What the neighbours of a macroblock need from it.
#[derive(Clone, Copy)]
struct MbInfo {
    /// Intra_4x4 modes by raster 4x4 position, 2 (DC) for an Intra_16x16 macroblock.
    modes: [u8; 16],
    /// TotalCoeff of each luma 4x4 block (raster), and of the chroma 4x4 blocks.
    nz: [u8; 16],
    nz_c: [[u8; 8]; 2],
}

const NO_MB: MbInfo = MbInfo { modes: [2; 16], nz: [0; 16], nz_c: [[0; 8]; 2] };

/// The coded content of a macroblock, before writing.
struct MbCode {
    /// Intra_16x16 mode, or `None` for Intra_4x4.
    i16: Option<u8>,
    /// Intra_4x4: (prev_intra4x4_pred_mode_flag, rem) for the 16 blocks in decoding order.
    i4: [(bool, u8); 16],
    chroma_mode: u8,
    cbp_l: u8,
    cbp_c: u8,
    /// Luma levels by raster 4x4 block (raster coefficients); AC only for Intra_16x16.
    luma: [[i32; 16]; 16],
    /// Intra_16x16 DC levels (raster of the blocks).
    luma_dc: [i32; 16],
    /// Chroma AC levels [component][block][raster coefficient], DC [component][row][column].
    chroma: [[[i32; 16]; 8]; 2],
    chroma_dc: [[[i32; 2]; 4]; 2],
}

/// The reconstructed samples of the slice (its rows only: nothing above it is available).
struct Recon {
    y: Vec<i32>,
    c: [Vec<i32>; 2],
    w: usize,
}

impl Recon {
    fn luma(&self, x: usize, y: usize) -> i32 {
        self.y[y * self.w + x]
    }
    fn chroma(&self, k: usize, x: usize, y: usize) -> i32 {
        self.c[k][y * (self.w / 2) + x]
    }
}

struct Ctx<'a> {
    src: &'a Planes<'a>,
    /// First luma row of the slice in the picture.
    y0: usize,
    rec: Recon,
    qp: i32,
    qpc: i32,
    lambda: f64,
    /// The scaling of the luma and of the chroma.
    sl: &'a Scale,
    sc: &'a Scale,
}

/// Code macroblock pair rows `r0..r1` of a frame as one IDR slice NAL unit, every pair a field
/// macroblock pair (MBAFF, `mb_field_decoding_flag` 1): the top macroblock from the top field,
/// the bottom one from the bottom field. With field pairs all around, a field macroblock's
/// neighbours (Table 6-4) are the same-parity macroblocks of the neighbouring pairs, so each
/// field is predicted as a field picture would be. QP_Y `qp` (−12..51 for 10-bit video).
pub fn encode_slice(fields: [&Planes; 2], r0: usize, r1: usize, qp: i32, idr_pic_id: u32) -> Vec<u8> {
    let (sl, sc) = (Scale::new(weights(&DEFAULT_4X4_INTRA)), Scale::new(weights(&CHROMA_4X4)));
    let w = fields[0].width;
    let rows = (r1 - r0) * 16;
    let qp_prime = qp + 12;
    // chroma: qPI from QP_Y and chroma_qp_index_offset, QP′C = QPC + QpBdOffsetC
    let qpc_prime = qpc((qp + CHROMA_QP_OFFSET).clamp(-12, 51)) + 12;
    let lambda = 4.0 * (0.85 * 2f64.powf(f64::from(qp - 12) / 3.0)).sqrt();
    let mut ctx = fields.map(|src| Ctx {
        src,
        y0: r0 * 16,
        rec: Recon { y: vec![0; w * rows], c: [vec![0; w / 2 * rows], vec![0; w / 2 * rows]], w },
        qp: qp_prime,
        qpc: qpc_prime,
        lambda,
        sl: &sl,
        sc: &sc,
    });
    let mut bw = BitWriter::new();
    slice_header(&mut bw, r0 * MB_W, idr_pic_id, qp);
    let mb_w = w / 16;
    let mut above = [vec![NO_MB; mb_w], vec![NO_MB; mb_w]];
    let mut cur = [vec![NO_MB; mb_w], vec![NO_MB; mb_w]];
    for row in 0..(r1 - r0) {
        for mbx in 0..mb_w {
            bw.write_bit(true); // mb_field_decoding_flag of the pair
            for f in 0..2 {
                let left = (mbx > 0).then(|| cur[f][mbx - 1]);
                let top = (row > 0).then(|| above[f][mbx]);
                let nb = Nb { left, top, has_tr: row > 0 && mbx + 1 < mb_w, has_tl: row > 0 && mbx > 0 };
                let (code, info) = ctx[f].code_mb(mbx * 16, row * 16, &nb);
                write_mb(&mut bw, &code, &info, &nb);
                cur[f][mbx] = info;
            }
        }
        std::mem::swap(&mut above, &mut cur);
    }
    bw.rbsp_trailing();
    nal(3, NAL_IDR, &bw.finish())
}

/// The neighbouring macroblocks (when available, in the slice).
struct Nb {
    left: Option<MbInfo>,
    top: Option<MbInfo>,
    has_tr: bool,
    has_tl: bool,
}

impl Ctx<'_> {
    fn src_luma(&self, x: usize, y: usize) -> i32 {
        i32::from(self.src.y[(self.y0 + y) * self.src.width + x])
    }
    fn src_chroma(&self, k: usize, x: usize, y: usize) -> i32 {
        let p = if k == 0 { self.src.cb } else { self.src.cr };
        i32::from(p[(self.y0 + y) * (self.src.width / 2) + x])
    }

    /// Code the macroblock whose top-left luma sample is (`x`, `y`) in the slice.
    fn code_mb(&mut self, x: usize, y: usize, nb: &Nb) -> (MbCode, MbInfo) {
        let mut code = MbCode {
            i16: None,
            i4: [(true, 0); 16],
            chroma_mode: 0,
            cbp_l: 0,
            cbp_c: 0,
            luma: [[0; 16]; 16],
            luma_dc: [0; 16],
            chroma: [[[0; 16]; 8]; 2],
            chroma_dc: [[[0; 2]; 4]; 2],
        };
        let mut info = NO_MB;
        let src: [i32; 256] = std::array::from_fn(|i| self.src_luma(x + i % 16, y + i / 16));

        // Intra_16x16: the best mode by SATD
        let e16 = self.edge_luma(x, y, nb);
        let mut best16 = (u64::MAX, 0u8);
        let mut p16 = [0i32; 256];
        for m in 0..4u8 {
            if !mode16_available(m, &e16) {
                continue;
            }
            pred16x16(m, &e16, &mut p16);
            let mut satd = 0u64;
            for b in 0..16 {
                let (bx, by) = (b % 4, b / 4);
                let d: [i32; 16] = std::array::from_fn(|i| src[(by * 4 + i / 4) * 16 + bx * 4 + i % 4] - p16[(by * 4 + i / 4) * 16 + bx * 4 + i % 4]);
                satd += u64::from(satd4(&d));
            }
            let cost = satd + (self.lambda * 4.0) as u64;
            if cost < best16.0 {
                best16 = (cost, m);
            }
        }

        // Intra_4x4, block by block on the reconstruction (it predicts the next blocks)
        let mut cost4 = (self.lambda * 2.0) as u64;
        let mut levels4 = [[0i32; 16]; 16];
        let mut modes4 = [2u8; 16];
        let mut flags4 = [(true, 0u8); 16];
        for blk in 0..16 {
            let (bx, by) = BLK4_XY[blk];
            let e = self.edge4(x, y, bx, by, nb);
            let predicted = self.predicted_mode(&modes4, bx, by, nb);
            let s: [i32; 16] = std::array::from_fn(|i| src[(by * 4 + i / 4) * 16 + bx * 4 + i % 4]);
            let mut best = (u64::MAX, 2u8, [0i32; 16]);
            let mut p = [0i32; 16];
            for m in 0..9u8 {
                if !mode4_available(m, &e) {
                    continue;
                }
                pred4x4(m, &e, &mut p);
                let d: [i32; 16] = std::array::from_fn(|i| s[i] - p[i]);
                let bits = if m == predicted { 1.0 } else { 4.0 };
                let cost = u64::from(satd4(&d)) + (self.lambda * bits) as u64;
                if cost < best.0 {
                    best = (cost, m, p);
                }
            }
            cost4 += best.0;
            let (m, p) = (best.1, best.2);
            modes4[by * 4 + bx] = m;
            flags4[blk] = if m == predicted { (true, 0) } else { (false, if m < predicted { m } else { m - 1 }) };
            let d: [i32; 16] = std::array::from_fn(|i| s[i] - p[i]);
            let mut c = fdct4(&d);
            quant4(&mut c, self.sl, self.qp, false);
            levels4[by * 4 + bx] = c;
            let mut dq = c;
            dequant4(&mut dq, self.sl, self.qp, false);
            let r = idct4(&dq);
            for i in 0..16 {
                self.rec.y[(y + by * 4 + i / 4) * self.rec.w + x + bx * 4 + i % 4] = (p[i] + r[i]).clamp(0, MAX);
            }
        }

        if best16.0 < cost4 {
            // Intra_16x16 wins: code it over the Intra_4x4 reconstruction
            let m = best16.1;
            pred16x16(m, &e16, &mut p16);
            let mut coef = [[0i32; 16]; 16];
            let mut dc = [0i32; 16];
            for b in 0..16 {
                let (bx, by) = (b % 4, b / 4);
                let d: [i32; 16] = std::array::from_fn(|i| src[(by * 4 + i / 4) * 16 + bx * 4 + i % 4] - p16[(by * 4 + i / 4) * 16 + bx * 4 + i % 4]);
                coef[b] = fdct4(&d);
                dc[b] = coef[b][0];
                quant4(&mut coef[b], self.sl, self.qp, true);
                coef[b][0] = 0;
            }
            let dc_lv = quant_luma_dc(&dc, self.sl, self.qp);
            let dc_rec = dequant_luma_dc(&dc_lv, self.sl, self.qp);
            let any_ac = coef.iter().any(|c| c.iter().any(|&v| v != 0));
            code.cbp_l = if any_ac { 15 } else { 0 };
            for b in 0..16 {
                let (bx, by) = (b % 4, b / 4);
                let mut dq = coef[b];
                dequant4(&mut dq, self.sl, self.qp, true);
                dq[0] = dc_rec[b];
                let r = idct4(&dq);
                for i in 0..16 {
                    let (px, py) = (bx * 4 + i % 4, by * 4 + i / 4);
                    self.rec.y[(y + py) * self.rec.w + x + px] = (p16[py * 16 + px] + r[i]).clamp(0, MAX);
                }
                info.nz[b] = if any_ac { coef[b].iter().filter(|&&v| v != 0).count() as u8 } else { 0 };
            }
            code.i16 = Some(m);
            code.luma = coef;
            code.luma_dc = dc_lv;
            info.modes = [2; 16];
        } else {
            code.i4 = flags4;
            code.luma = levels4;
            for b8 in 0..4 {
                let (ox, oy) = ((b8 % 2) * 2, (b8 / 2) * 2);
                let coded = (0..4).any(|k| levels4[(oy + k / 2) * 4 + ox + k % 2].iter().any(|&v| v != 0));
                if coded {
                    code.cbp_l |= 1 << b8;
                }
                for k in 0..4 {
                    let r = (oy + k / 2) * 4 + ox + k % 2;
                    info.nz[r] = if coded { levels4[r].iter().filter(|&&v| v != 0).count() as u8 } else { 0 };
                }
            }
            info.modes = modes4;
        }

        self.code_chroma(x / 2, y, nb, &mut code, &mut info);
        (code, info)
    }

    /// predIntra4x4PredMode (8.3.1.1) for the block at (`bx`, `by`).
    fn predicted_mode(&self, modes: &[u8; 16], bx: usize, by: usize, nb: &Nb) -> u8 {
        let a = if bx > 0 { Some(modes[by * 4 + bx - 1]) } else { nb.left.map(|m| m.modes[by * 4 + 3]) };
        let b = if by > 0 { Some(modes[(by - 1) * 4 + bx]) } else { nb.top.map(|m| m.modes[12 + bx]) };
        match (a, b) {
            (Some(a), Some(b)) => a.min(b),
            _ => 2,
        }
    }

    fn edge4(&self, x: usize, y: usize, bx: usize, by: usize, nb: &Nb) -> Edge4 {
        let (px, py) = (x + bx * 4, y + by * 4);
        let has_top = by > 0 || nb.top.is_some();
        let has_left = bx > 0 || nb.left.is_some();
        let has_tl = match (bx > 0, by > 0) {
            (true, true) => true,
            (false, true) => nb.left.is_some(),
            (true, false) => nb.top.is_some(),
            (false, false) => nb.has_tl,
        };
        // top-right: inside the macroblock only where already coded; the top row's from above
        let blk = crate::tables::RASTER_TO_BLK4[by * 4 + bx];
        let has_tr = if by == 0 {
            if bx < 3 { nb.top.is_some() } else { nb.has_tr }
        } else {
            !matches!(blk, 3 | 7 | 11 | 13 | 15)
        };
        let mut top = [0i32; 8];
        if has_top {
            for i in 0..4 {
                top[i] = self.rec.luma(px + i, py - 1);
            }
            for i in 4..8 {
                top[i] = if has_tr { self.rec.luma(px + i, py - 1) } else { top[3] };
            }
        }
        let mut left = [0i32; 4];
        if has_left {
            for i in 0..4 {
                left[i] = self.rec.luma(px - 1, py + i);
            }
        }
        let tl = if has_tl { self.rec.luma(px - 1, py - 1) } else { 0 };
        Edge4 { top, left, tl, has_top, has_left, has_tl }
    }

    fn edge_luma(&self, x: usize, y: usize, nb: &Nb) -> EdgeLuma {
        let (has_top, has_left) = (nb.top.is_some(), nb.left.is_some());
        let top = std::array::from_fn(|i| if has_top { self.rec.luma(x + i, y - 1) } else { 0 });
        let left = std::array::from_fn(|i| if has_left { self.rec.luma(x - 1, y + i) } else { 0 });
        let tl = if nb.has_tl { self.rec.luma(x - 1, y - 1) } else { 0 };
        EdgeLuma { top, left, tl, has_top, has_left, has_tl: nb.has_tl }
    }

    fn edge_chroma(&self, k: usize, cx: usize, y: usize, nb: &Nb) -> EdgeChroma {
        let (has_top, has_left) = (nb.top.is_some(), nb.left.is_some());
        let top = std::array::from_fn(|i| if has_top { self.rec.chroma(k, cx + i, y - 1) } else { 0 });
        let left = std::array::from_fn(|i| if has_left { self.rec.chroma(k, cx - 1, y + i) } else { 0 });
        let tl = if nb.has_tl { self.rec.chroma(k, cx - 1, y - 1) } else { 0 };
        EdgeChroma { top, left, tl, has_top, has_left, has_tl: nb.has_tl }
    }

    /// Both chroma blocks (8x16 each at chroma column `cx`): the mode by SATD, then coding.
    fn code_chroma(&mut self, cx: usize, y: usize, nb: &Nb, code: &mut MbCode, info: &mut MbInfo) {
        let edges = [self.edge_chroma(0, cx, y, nb), self.edge_chroma(1, cx, y, nb)];
        let src: [[i32; 128]; 2] = std::array::from_fn(|k| std::array::from_fn(|i| self.src_chroma(k, cx + i % 8, y + i / 8)));
        let mut best = (u64::MAX, 0u8);
        let mut p = [0i32; 128];
        for m in 0..4u8 {
            if !mode_chroma_available(m, &edges[0]) {
                continue;
            }
            let mut satd = 0u64;
            for k in 0..2 {
                pred_chroma422(m, &edges[k], &mut p);
                for b in 0..8 {
                    let (bx, by) = ((b % 2) * 4, (b / 2) * 4);
                    let d: [i32; 16] = std::array::from_fn(|i| src[k][(by + i / 4) * 8 + bx + i % 4] - p[(by + i / 4) * 8 + bx + i % 4]);
                    satd += u64::from(satd4(&d));
                }
            }
            let bits = [1.0, 3.0, 3.0, 5.0][m as usize];
            let cost = satd + (self.lambda * bits) as u64;
            if cost < best.0 {
                best = (cost, m);
            }
        }
        code.chroma_mode = best.1;
        let mut preds = [[0i32; 128]; 2];
        let mut any_ac = false;
        let mut any_dc = false;
        for k in 0..2 {
            pred_chroma422(best.1, &edges[k], &mut preds[k]);
            let mut dc = [[0i32; 2]; 4];
            for b in 0..8 {
                let (bx, by) = ((b % 2) * 4, (b / 2) * 4);
                let d: [i32; 16] = std::array::from_fn(|i| src[k][(by + i / 4) * 8 + bx + i % 4] - preds[k][(by + i / 4) * 8 + bx + i % 4]);
                let mut c = fdct4(&d);
                dc[b / 2][b % 2] = c[0];
                quant4(&mut c, self.sc, self.qpc, true);
                c[0] = 0;
                any_ac |= c.iter().any(|&v| v != 0);
                code.chroma[k][b] = c;
            }
            code.chroma_dc[k] = quant_chroma_dc422(&dc, self.sc, self.qpc);
            any_dc |= code.chroma_dc[k].iter().flatten().any(|&v| v != 0);
        }
        code.cbp_c = if any_ac {
            2
        } else if any_dc {
            1
        } else {
            0
        };
        for k in 0..2 {
            let dc_rec = dequant_chroma_dc422(&code.chroma_dc[k], self.sc, self.qpc);
            for b in 0..8 {
                let (bx, by) = ((b % 2) * 4, (b / 2) * 4);
                let mut dq = code.chroma[k][b];
                dequant4(&mut dq, self.sc, self.qpc, true);
                dq[0] = dc_rec[b / 2][b % 2];
                let r = idct4(&dq);
                for i in 0..16 {
                    let (px, py) = (bx + i % 4, by + i / 4);
                    self.rec.c[k][(y + py) * (self.rec.w / 2) + cx + px] = (preds[k][py * 8 + px] + r[i]).clamp(0, MAX);
                }
                info.nz_c[k][b] = if code.cbp_c == 2 { code.chroma[k][b].iter().filter(|&&v| v != 0).count() as u8 } else { 0 };
            }
        }
    }
}

/// Write a macroblock layer (CAVLC, I slice).
fn write_mb(w: &mut BitWriter, c: &MbCode, info: &MbInfo, nb: &Nb) {
    match c.i16 {
        Some(m) => w.write_ue(1 + u32::from(m) + 4 * u32::from(c.cbp_c) + if c.cbp_l == 15 { 12 } else { 0 }),
        None => {
            w.write_ue(0); // I_NxN
            w.write_bit(false); // transform_size_8x8_flag: 4x4
            for &(flag, rem) in &c.i4 {
                w.write_bit(flag);
                if !flag {
                    w.write_bits(u32::from(rem), 3);
                }
            }
        }
    }
    w.write_ue(u32::from(c.chroma_mode));
    if c.i16.is_none() {
        w.write_ue(u32::from(CBP_INTRA_TO_CODE[usize::from(c.cbp_l | (c.cbp_c << 4))]));
    }
    if c.i16.is_some() || c.cbp_l != 0 || c.cbp_c != 0 {
        w.write_se(0); // mb_qp_delta
    } else {
        return;
    }
    // luma
    let nc_luma = |bx: usize, by: usize| {
        let a = if bx > 0 { Some(info.nz[by * 4 + bx - 1]) } else { nb.left.map(|m| m.nz[by * 4 + 3]) };
        let b = if by > 0 { Some(info.nz[(by - 1) * 4 + bx]) } else { nb.top.map(|m| m.nz[12 + bx]) };
        nc_of(a, b)
    };
    if c.i16.is_some() {
        let dc: [i32; 16] = std::array::from_fn(|k| c.luma_dc[FIELD4[k]]);
        write_residual(w, &dc, nc_luma(0, 0));
        if c.cbp_l == 15 {
            for blk in 0..16 {
                let (bx, by) = BLK4_XY[blk];
                let l = &c.luma[by * 4 + bx];
                let ac: [i32; 15] = std::array::from_fn(|k| l[FIELD4[k + 1]]);
                write_residual(w, &ac, nc_luma(bx, by));
            }
        }
    } else {
        for blk in 0..16 {
            if c.cbp_l & (1 << (blk / 4)) == 0 {
                continue;
            }
            let (bx, by) = BLK4_XY[blk];
            let l = &c.luma[by * 4 + bx];
            let coeffs: [i32; 16] = std::array::from_fn(|k| l[FIELD4[k]]);
            write_residual(w, &coeffs, nc_luma(bx, by));
        }
    }
    // chroma: both DC, then both components' AC blocks
    if c.cbp_c != 0 {
        for k in 0..2 {
            let dc: [i32; 8] = std::array::from_fn(|i| c.chroma_dc[k][CDC422_POS[i].0][CDC422_POS[i].1]);
            write_residual(w, &dc, NC_CDC422);
        }
    }
    if c.cbp_c == 2 {
        for k in 0..2 {
            for b in 0..8 {
                let a = if b % 2 == 1 { Some(info.nz_c[k][b - 1]) } else { nb.left.map(|m| m.nz_c[k][b + 1]) };
                let t = if b >= 2 { Some(info.nz_c[k][b - 2]) } else { nb.top.map(|m| m.nz_c[k][b + 6]) };
                let l = &c.chroma[k][b];
                let ac: [i32; 15] = std::array::from_fn(|i| l[FIELD4[i + 1]]);
                write_residual(w, &ac, nc_of(a, t));
            }
        }
    }
}
