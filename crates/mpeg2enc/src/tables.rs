//! Encoding lookups built from the H.262 Annex B code lists in `vlc_data.rs`, the quantiser
//! scales, scans and the default intra matrix.

use std::sync::OnceLock;

/// A code: its bits (right-aligned) and length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Code {
    pub bits: u32,
    pub len: u8,
}

mod data {
    include!("vlc_data.rs");

    /// Table zero (B.14) and table one (B.15), each with the shared long codes.
    pub fn dct(one: bool) -> Vec<(&'static str, i16)> {
        let mut v = if one { DCT_ONE.to_vec() } else { [DCT_ZERO, DCT_ZERO_LONG].concat() };
        v.extend_from_slice(DCT_MID_SHARED);
        v.extend_from_slice(DCT_LONG);
        v
    }
}

fn parse(s: &str) -> Code {
    let mut c = Code::default();
    for b in s.bytes().filter(|b| *b == b'0' || *b == b'1') {
        c.bits = (c.bits << 1) | u32::from(b - b'0');
        c.len += 1;
    }
    c
}

/// Longest run and level with a code of their own (escape beyond).
const MAX_RUN: usize = 32;
const MAX_LEVEL: usize = 41;

pub struct Tables {
    /// macroblock_address_increment 1..=33 (index), escape at 34.
    pub mba: [Code; 35],
    /// coded_block_pattern (6-bit value) → code.
    pub cbp: [Code; 64],
    /// motion_code −16..=16 at index + 16.
    pub motion: [Code; 33],
    pub dc_luma: [Code; 12],
    pub dc_chroma: [Code; 12],
    /// [table][run][level]: the code without its sign bit (len 0: none, escape).
    pub dct: [Box<[[Code; MAX_LEVEL]; MAX_RUN]>; 2],
    pub eob: [Code; 2],
    pub escape: Code,
}

pub fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            mba: [Code::default(); 35],
            cbp: [Code::default(); 64],
            motion: [Code::default(); 33],
            dc_luma: [Code::default(); 12],
            dc_chroma: [Code::default(); 12],
            dct: [Box::new([[Code::default(); MAX_LEVEL]; MAX_RUN]), Box::new([[Code::default(); MAX_LEVEL]; MAX_RUN])],
            eob: [Code::default(); 2],
            escape: parse("0000 01"),
        };
        for &(s, v) in data::MBA {
            if (1..=34).contains(&v) {
                t.mba[v as usize] = parse(s);
            }
        }
        for &(s, v) in data::CBP {
            t.cbp[v as usize] = parse(s);
        }
        for &(s, v) in data::MOTION {
            t.motion[(v + 16) as usize] = parse(s);
        }
        for &(s, v) in data::DC_LUMA {
            t.dc_luma[v as usize] = parse(s);
        }
        for &(s, v) in data::DC_CHROMA {
            t.dc_chroma[v as usize] = parse(s);
        }
        for one in [false, true] {
            for (s, v) in data::dct(one) {
                let k = usize::from(one);
                match v {
                    -1 => t.eob[k] = parse(s),
                    -2 => {}
                    v => t.dct[k][(v & 63) as usize][(v >> 6) as usize] = parse(s),
                }
            }
        }
        t
    })
}

impl Tables {
    /// The code of (run, |level|) in table `one`, if it has one.
    pub fn run_level(&self, one: bool, run: usize, level: u32) -> Option<Code> {
        let c = *self.dct[usize::from(one)].get(run)?.get(level as usize)?;
        (c.len > 0).then_some(c)
    }
}

/// Non-linear quantiser_scale (Table 7-6), indexed by quantiser_scale_code.
pub const NON_LINEAR_Q: [i32; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 88, 96, 104, 112];

/// The quantiser_scale_code whose scale is nearest `q`.
pub fn q_code(q: f64) -> usize {
    (1..32).min_by(|&a, &b| (f64::from(NON_LINEAR_Q[a]) - q).abs().total_cmp(&(f64::from(NON_LINEAR_Q[b]) - q).abs())).unwrap_or(1)
}

/// Default intra quantiser matrix (H.262 §6.3.11), raster order.
pub const DEFAULT_INTRA: [i32; 64] = [
    8, 16, 19, 22, 26, 27, 29, 34, //
    16, 16, 22, 24, 27, 29, 34, 37, //
    19, 22, 26, 27, 29, 34, 34, 38, //
    22, 22, 26, 27, 29, 34, 37, 40, //
    22, 26, 27, 29, 32, 35, 40, 48, //
    26, 27, 29, 32, 35, 40, 48, 58, //
    26, 27, 29, 34, 38, 46, 56, 69, //
    27, 29, 35, 38, 46, 56, 69, 83,
];

/// Alternate scan (H.262 Figure 7-3): scan index → raster position.
pub const ALT_SCAN: [usize; 64] = [
    0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3, 11, 4, 12, 19, 27, 34, 42, 50, 58, 35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29,
    36, 44, 52, 60, 37, 45, 53, 61, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_is_filled() {
        let t = tables();
        assert!(t.mba[1..=34].iter().all(|c| c.len > 0));
        assert!(t.cbp.iter().all(|c| c.len > 0));
        assert!(t.motion.iter().all(|c| c.len > 0));
        assert!(t.dc_luma.iter().chain(&t.dc_chroma).all(|c| c.len > 0));
        assert_eq!(t.eob[0], Code { bits: 0b10, len: 2 });
        assert_eq!(t.eob[1], Code { bits: 0b0110, len: 4 });
        // 111 run/level pairs in each table
        for k in 0..2 {
            assert_eq!(t.dct[k].iter().flatten().filter(|c| c.len > 0).count(), 111);
        }
        assert_eq!(t.run_level(false, 0, 1), Some(Code { bits: 0b11, len: 2 }));
        assert_eq!(t.run_level(true, 0, 15), Some(Code { bits: 0b1111_1111, len: 8 }));
        assert_eq!(t.run_level(false, 31, 2), None);
    }
}
