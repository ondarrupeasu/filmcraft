//! CAVLC residual block coding (§9.2), with the 4:2:2 chroma DC tables (nC = −2) and the
//! level_prefix escapes of the High profiles (levels of 10-bit video).

use filmcraft_bitstream::BitWriter;

use crate::tables::{COEFF_TOKEN_CDC422, COEFF_TOKEN_CODE, COEFF_TOKEN_LEN, RUN_BEFORE, TOTAL_ZEROS, TOTAL_ZEROS_CDC422};

/// nC of the 4:2:2 chroma DC block.
pub const NC_CDC422: i32 = -2;

/// Write one residual block (levels in scan order, `coeffs.len()` = maxNumCoeff); `nc` is the
/// predicted coefficient count, or [`NC_CDC422`]. Returns TotalCoeff.
pub fn write_residual(w: &mut BitWriter, coeffs: &[i32], nc: i32) -> u8 {
    let max = coeffs.len();
    // levels from the highest frequency down
    let mut levels = [0i32; 16];
    let mut pos = [0usize; 16];
    let mut total = 0usize;
    for i in (0..max).rev() {
        if coeffs[i] != 0 {
            levels[total] = coeffs[i];
            pos[total] = i;
            total += 1;
        }
    }
    let mut t1 = 0usize;
    while t1 < total && t1 < 3 && levels[t1].abs() == 1 {
        t1 += 1;
    }
    if nc == NC_CDC422 {
        let (len, code) = COEFF_TOKEN_CDC422[total][t1];
        w.write_bits(u32::from(code), u32::from(len));
    } else if nc >= 8 {
        w.write_bits(if total == 0 { 3 } else { (((total - 1) << 2) | t1) as u32 }, 6);
    } else {
        let t = if nc < 2 {
            0
        } else if nc < 4 {
            1
        } else {
            2
        };
        w.write_bits(u32::from(COEFF_TOKEN_CODE[t][total][t1]), u32::from(COEFF_TOKEN_LEN[t][total][t1]));
    }
    if total == 0 {
        return 0;
    }
    let mut suffix_len: u32 = if total > 10 && t1 < 3 { 1 } else { 0 };
    for i in 0..total {
        let l = levels[i];
        if i < t1 {
            w.write_bit(l < 0);
            continue;
        }
        // levelCode (inverse of 9.2.2 step 8), less 2 for the first after fewer than 3 trailing ones
        let mut code = if l > 0 { 2 * i64::from(l) - 2 } else { -2 * i64::from(l) - 1 };
        if i == t1 && t1 < 3 {
            code -= 2;
        }
        let code = code as u64;
        if suffix_len == 0 {
            if code < 14 {
                w.write_bits(1, code as u32 + 1);
            } else if code < 30 {
                // level_prefix 14 with its 4-bit suffix
                w.write_bits(1, 15);
                w.write_bits((code - 14) as u32, 4);
            } else {
                write_escape(w, code - 30);
            }
        } else if code < (15u64 << suffix_len) {
            w.write_bits(1, (code >> suffix_len) as u32 + 1);
            w.write_bits((code & ((1 << suffix_len) - 1)) as u32, suffix_len);
        } else {
            write_escape(w, code - (15u64 << suffix_len));
        }
        if suffix_len == 0 {
            suffix_len = 1;
        }
        if l.unsigned_abs() > (3 << (suffix_len - 1)) && suffix_len < 6 {
            suffix_len += 1;
        }
    }
    let total_zeros = pos[0] + 1 - total;
    if total < max {
        let (len, code) = if nc == NC_CDC422 { TOTAL_ZEROS_CDC422[total - 1][total_zeros] } else { TOTAL_ZEROS[total - 1][total_zeros] };
        w.write_bits(u32::from(code), u32::from(len));
    }
    let mut zeros_left = total_zeros;
    for i in 0..total - 1 {
        if zeros_left == 0 {
            break;
        }
        let run = pos[i] - pos[i + 1] - 1;
        let (len, code) = RUN_BEFORE[zeros_left.min(7) - 1][run];
        w.write_bits(u32::from(code), u32::from(len));
        zeros_left -= run;
    }
    total as u8
}

/// level_prefix ≥ 15: `rem` is levelCode less the base of prefix 15 (9.2.2 steps 4–6).
fn write_escape(w: &mut BitWriter, rem: u64) {
    if rem < 4096 {
        w.write_bits(1, 16);
        w.write_bits(rem as u32, 12);
    } else {
        // level_prefix p ≥ 16: levelCode gains (1 << (p − 3)) − 4096, the suffix has p − 3 bits
        let mut p = 16u32;
        while rem + 4096 >= (1u64 << (p - 3)) * 2 {
            p += 1;
        }
        w.write_bits(0, p);
        w.write_bit(true);
        w.write_bits((rem + 4096 - (1u64 << (p - 3))) as u32, p - 3);
    }
}

/// nC from the neighbours' TotalCoeff (9.2.1).
pub fn nc_of(a: Option<u8>, b: Option<u8>) -> i32 {
    match (a, b) {
        (Some(a), Some(b)) => (i32::from(a) + i32::from(b) + 1) >> 1,
        (Some(a), None) => i32::from(a),
        (None, Some(b)) => i32::from(b),
        _ => 0,
    }
}
