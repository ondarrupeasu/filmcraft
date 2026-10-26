//! Intra prediction (§8.3) of 10-bit samples: Intra_4x4, Intra_16x16 and 4:2:2 chroma (8x16).

/// Samples are 10-bit: the DC value without neighbours and the clipping range.
pub const MID: i32 = 512;
pub const MAX: i32 = 1023;

/// The neighbours of a 4x4 block: `top` p[x,−1] for x = 0..7 (top-right already substituted by
/// p[3,−1] when not available), `left` p[−1,y], `tl` p[−1,−1].
#[derive(Clone, Copy)]
pub struct Edge4 {
    pub top: [i32; 8],
    pub left: [i32; 4],
    pub tl: i32,
    pub has_top: bool,
    pub has_left: bool,
    pub has_tl: bool,
}

/// Can Intra_4x4 mode `mode` be used with these neighbours?
pub fn mode4_available(mode: u8, e: &Edge4) -> bool {
    match mode {
        0 | 3 | 7 => e.has_top,
        1 | 8 => e.has_left,
        2 => true,
        _ => e.has_top && e.has_left && e.has_tl,
    }
}

/// Intra_4x4 prediction (§8.3.1.2) into raster `out`.
pub fn pred4x4(mode: u8, e: &Edge4, out: &mut [i32; 16]) {
    let dc = match (e.has_top, e.has_left) {
        (true, true) => (e.top[..4].iter().sum::<i32>() + e.left.iter().sum::<i32>() + 4) >> 3,
        (true, false) => (e.top[..4].iter().sum::<i32>() + 2) >> 2,
        (false, true) => (e.left.iter().sum::<i32>() + 2) >> 2,
        _ => MID,
    };
    if mode == 2 {
        out.fill(dc);
        return;
    }
    let t = |x: i32| if x < 0 { e.tl } else { e.top[x as usize] };
    let l = |y: i32| if y < 0 { e.tl } else { e.left[y as usize] };
    for y in 0..4i32 {
        for x in 0..4i32 {
            let v = match mode {
                0 => t(x),
                1 => l(y),
                3 => {
                    if x == 3 && y == 3 {
                        (t(6) + 3 * t(7) + 2) >> 2
                    } else {
                        (t(x + y) + 2 * t(x + y + 1) + t(x + y + 2) + 2) >> 2
                    }
                }
                4 => {
                    if x > y {
                        (t(x - y - 2) + 2 * t(x - y - 1) + t(x - y) + 2) >> 2
                    } else if x < y {
                        (l(y - x - 2) + 2 * l(y - x - 1) + l(y - x) + 2) >> 2
                    } else {
                        (t(0) + 2 * t(-1) + l(0) + 2) >> 2
                    }
                }
                5 => {
                    let z = 2 * x - y;
                    if z >= 0 && z & 1 == 0 {
                        (t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (t(x - (y >> 1) - 2) + 2 * t(x - (y >> 1) - 1) + t(x - (y >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * l(-1) + t(0) + 2) >> 2
                    } else {
                        (l(y - 1) + 2 * l(y - 2) + l(y - 3) + 2) >> 2
                    }
                }
                6 => {
                    let z = 2 * y - x;
                    if z >= 0 && z & 1 == 0 {
                        (l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 1) >> 1
                    } else if z >= 0 {
                        (l(y - (x >> 1) - 2) + 2 * l(y - (x >> 1) - 1) + l(y - (x >> 1)) + 2) >> 2
                    } else if z == -1 {
                        (l(0) + 2 * l(-1) + t(0) + 2) >> 2
                    } else {
                        (t(x - 1) + 2 * t(x - 2) + t(x - 3) + 2) >> 2
                    }
                }
                7 => {
                    if y & 1 == 0 {
                        (t(x + (y >> 1)) + t(x + (y >> 1) + 1) + 1) >> 1
                    } else {
                        (t(x + (y >> 1)) + 2 * t(x + (y >> 1) + 1) + t(x + (y >> 1) + 2) + 2) >> 2
                    }
                }
                _ => {
                    let z = x + 2 * y;
                    if z < 5 && z & 1 == 0 {
                        (l(y + (x >> 1)) + l(y + (x >> 1) + 1) + 1) >> 1
                    } else if z < 5 {
                        (l(y + (x >> 1)) + 2 * l(y + (x >> 1) + 1) + l(y + (x >> 1) + 2) + 2) >> 2
                    } else if z == 5 {
                        (l(2) + 3 * l(3) + 2) >> 2
                    } else {
                        l(3)
                    }
                }
            };
            out[(y * 4 + x) as usize] = v;
        }
    }
}

/// The neighbours of a macroblock's 16x16 luma, or of its 8x16 chroma (`N` = 16 top samples for
/// luma, 8 for chroma; `M` = 16 left samples).
#[derive(Clone, Copy)]
pub struct EdgeMb<const N: usize, const M: usize> {
    pub top: [i32; N],
    pub left: [i32; M],
    pub tl: i32,
    pub has_top: bool,
    pub has_left: bool,
    pub has_tl: bool,
}

pub type EdgeLuma = EdgeMb<16, 16>;
pub type EdgeChroma = EdgeMb<8, 16>;

/// Intra_16x16 modes 0 V, 1 H, 2 DC, 3 plane: usable with these neighbours?
pub fn mode16_available(mode: u8, e: &EdgeLuma) -> bool {
    match mode {
        0 => e.has_top,
        1 => e.has_left,
        2 => true,
        _ => e.has_top && e.has_left && e.has_tl,
    }
}

/// Intra_16x16 prediction (§8.3.3) into raster `out`.
pub fn pred16x16(mode: u8, e: &EdgeLuma, out: &mut [i32; 256]) {
    match mode {
        0 => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].copy_from_slice(&e.top);
            }
        }
        1 => {
            for y in 0..16 {
                out[y * 16..y * 16 + 16].fill(e.left[y]);
            }
        }
        2 => {
            let (st, sl): (i32, i32) = (e.top.iter().sum(), e.left.iter().sum());
            let dc = match (e.has_top, e.has_left) {
                (true, true) => (st + sl + 16) >> 5,
                (true, false) => (st + 8) >> 4,
                (false, true) => (sl + 8) >> 4,
                _ => MID,
            };
            out.fill(dc);
        }
        _ => {
            let t = |x: i32| if x < 0 { e.tl } else { e.top[x as usize] };
            let l = |y: i32| if y < 0 { e.tl } else { e.left[y as usize] };
            let (mut h, mut v) = (0, 0);
            for i in 0..8 {
                h += (i + 1) * (t(8 + i) - t(6 - i));
                v += (i + 1) * (l(8 + i) - l(6 - i));
            }
            let a = 16 * (l(15) + t(15));
            let b = (5 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for y in 0..16 {
                for x in 0..16 {
                    out[y * 16 + x] = ((a + b * (x as i32 - 7) + c * (y as i32 - 7) + 16) >> 5).clamp(0, MAX);
                }
            }
        }
    }
}

/// Intra chroma modes 0 DC, 1 H, 2 V, 3 plane: usable with these neighbours?
pub fn mode_chroma_available(mode: u8, e: &EdgeChroma) -> bool {
    match mode {
        0 => true,
        1 => e.has_left,
        2 => e.has_top,
        _ => e.has_top && e.has_left && e.has_tl,
    }
}

/// 4:2:2 chroma prediction (§8.3.4) of an 8x16 block into raster `out` (8 wide).
pub fn pred_chroma422(mode: u8, e: &EdgeChroma, out: &mut [i32; 128]) {
    match mode {
        0 => {
            for blk in 0..8 {
                let (xo, yo) = ((blk % 2) * 4, (blk / 2) * 4);
                let st: i32 = e.top[xo..xo + 4].iter().sum();
                let sl: i32 = e.left[yo..yo + 4].iter().sum();
                let both = (st + sl + 4) >> 3;
                let (top, left) = ((st + 2) >> 2, (sl + 2) >> 2);
                let dc = if (xo == 0 && yo == 0) || (xo > 0 && yo > 0) {
                    match (e.has_top, e.has_left) {
                        (true, true) => both,
                        (false, true) => left,
                        (true, false) => top,
                        _ => MID,
                    }
                } else if xo > 0 {
                    // the top-row block on the right: the top first
                    if e.has_top {
                        top
                    } else if e.has_left {
                        left
                    } else {
                        MID
                    }
                } else if e.has_left {
                    left
                } else if e.has_top {
                    top
                } else {
                    MID
                };
                for y in 0..4 {
                    out[(yo + y) * 8 + xo..(yo + y) * 8 + xo + 4].fill(dc);
                }
            }
        }
        1 => {
            for y in 0..16 {
                out[y * 8..y * 8 + 8].fill(e.left[y]);
            }
        }
        2 => {
            for y in 0..16 {
                out[y * 8..y * 8 + 8].copy_from_slice(&e.top);
            }
        }
        _ => {
            // xCF = 0, yCF = 4 (equations 8-144 to 8-149)
            let t = |x: i32| if x < 0 { e.tl } else { e.top[x as usize] };
            let l = |y: i32| if y < 0 { e.tl } else { e.left[y as usize] };
            let (mut h, mut v) = (0, 0);
            for i in 0..4 {
                h += (i + 1) * (t(4 + i) - t(2 - i));
            }
            for i in 0..8 {
                v += (i + 1) * (l(8 + i) - l(6 - i));
            }
            let a = 16 * (l(15) + t(7));
            let b = (34 * h + 32) >> 6;
            let c = (5 * v + 32) >> 6;
            for y in 0..16 {
                for x in 0..8 {
                    out[y * 8 + x] = ((a + b * (x as i32 - 3) + c * (y as i32 - 7) + 16) >> 5).clamp(0, MAX);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_of_a_4x4() {
        let e = Edge4 { top: [100, 200, 300, 400, 400, 400, 400, 400], left: [10, 20, 30, 40], tl: 50, has_top: true, has_left: true, has_tl: true };
        let mut o = [0i32; 16];
        pred4x4(0, &e, &mut o);
        assert_eq!(&o[12..16], &[100, 200, 300, 400]);
        pred4x4(2, &e, &mut o);
        assert_eq!(o[0], (1000 + 100 + 4) >> 3);
        pred4x4(8, &e, &mut o);
        assert_eq!(o[15], 40);
        let none = Edge4 { has_top: false, has_left: false, has_tl: false, ..e };
        pred4x4(2, &none, &mut o);
        assert_eq!(o[0], MID);
    }

    #[test]
    fn chroma_dc_by_position() {
        let e = EdgeChroma { top: [100, 100, 100, 100, 300, 300, 300, 300], left: [8; 16], tl: 0, has_top: true, has_left: true, has_tl: true };
        let mut o = [0i32; 128];
        pred_chroma422(0, &e, &mut o);
        assert_eq!(o[0], (400 + 32 + 4) >> 3, "top-left block: both");
        assert_eq!(o[4], 300, "top-right block: the top");
        assert_eq!(o[8 * 4], 8, "left column below: the left");
        assert_eq!(o[8 * 4 + 4], (1200 + 32 + 4) >> 3, "inside: both");
    }
}
