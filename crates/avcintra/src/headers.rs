//! NAL units and parameter sets (§7.3): the High 4:2:2 Intra sequence of 1080 lines coded as
//! field pictures, its picture parameter set, access unit delimiters, slice headers and filler.

use filmcraft_bitstream::BitWriter;

use crate::tables::{CHROMA_4X4, LUMA_8X8};

/// NAL unit types used.
pub const NAL_IDR: u8 = 5;
const NAL_SPS: u8 = 7;
const NAL_PPS: u8 = 8;
const NAL_AUD: u8 = 9;
const NAL_FILLER: u8 = 12;

/// chroma_qp_index_offset (and second_chroma_qp_index_offset), as AVC-Intra streams have it.
pub const CHROMA_QP_OFFSET: i32 = 3;

/// Width in macroblocks, and height in macroblock pair rows (1080 lines coded as 1088; each
/// field 544).
pub const MB_W: usize = 120;
pub const MB_H: usize = 34;

/// One NAL unit with a 4-byte start code, its RBSP escaped (emulation prevention, §7.4.1).
pub fn nal(nal_ref_idc: u8, nal_type: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 6);
    out.extend_from_slice(&[0, 0, 0, 1, (nal_ref_idc << 5) | nal_type]);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
    }
    out
}

/// Sequence parameter set: High 4:2:2 Intra (profile_idc 122, constraint_set3_flag) level 4.1,
/// 4:2:2 10-bit, 1920x1080 interlaced as MBAFF frames (frame_mbs_only_flag 0,
/// mb_adaptive_frame_field_flag 1), BT.709 25 frames per second.
pub fn sps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(122, 8); // profile_idc
    w.write_bits(0b0001_0000, 8); // constraint_set3_flag (intra), reserved zero bits
    w.write_bits(41, 8); // level_idc
    w.write_ue(0); // seq_parameter_set_id
    w.write_ue(2); // chroma_format_idc 4:2:2
    w.write_ue(2); // bit_depth_luma_minus8
    w.write_ue(2); // bit_depth_chroma_minus8
    w.write_bit(false); // qpprime_y_zero_transform_bypass_flag
    w.write_bit(true); // seq_scaling_matrix_present_flag: RP 2027's lists
    for i in 0..8 {
        let list: Option<&[i32]> = match i {
            1 | 2 => Some(&CHROMA_4X4),
            6 => Some(&LUMA_8X8),
            _ => None,
        };
        w.write_bit(list.is_some()); // seq_scaling_list_present_flag[i]
        if let Some(list) = list {
            // delta_scale from the previous entry, modulo 256 (7.3.2.1.1.1)
            let mut last = 8;
            for &v in list {
                w.write_se((v - last + 128).rem_euclid(256) - 128);
                last = v;
            }
        }
    }
    w.write_ue(0); // log2_max_frame_num_minus4
    w.write_ue(0); // pic_order_cnt_type
    w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
    w.write_ue(0); // max_num_ref_frames (A.2.9)
    w.write_bit(false); // gaps_in_frame_num_value_allowed_flag
    w.write_ue((MB_W - 1) as u32); // pic_width_in_mbs_minus1
    w.write_ue((MB_H - 1) as u32); // pic_height_in_map_units_minus1 (macroblock pair rows)
    w.write_bit(false); // frame_mbs_only_flag
    w.write_bit(true); // mb_adaptive_frame_field_flag
    w.write_bit(true); // direct_8x8_inference_flag (required without frame_mbs_only)
    w.write_bit(true); // frame_cropping_flag: 1088 → 1080
    w.write_ue(0);
    w.write_ue(0);
    w.write_ue(0);
    w.write_ue(4); // frame_crop_bottom_offset in CropUnitY = SubHeightC · 2 = 2 lines
    w.write_bit(true); // vui_parameters_present_flag
    w.write_bit(true); // aspect_ratio_info_present_flag
    w.write_bits(1, 8); // 1:1
    w.write_bit(false); // overscan_info_present_flag
    w.write_bit(true); // video_signal_type_present_flag
    w.write_bits(1, 3); // video_format: PAL
    w.write_bit(false); // video_full_range_flag
    w.write_bit(true); // colour_description_present_flag
    w.write_bits(1, 8); // BT.709 primaries
    w.write_bits(1, 8); // BT.709 transfer
    w.write_bits(1, 8); // BT.709 matrix
    w.write_bit(false); // chroma_loc_info_present_flag
    w.write_bit(true); // timing_info_present_flag: a field per tick, 50 a second
    w.write_bits(1, 32);
    w.write_bits(50, 32);
    w.write_bit(true); // fixed_frame_rate_flag
    w.write_bit(false); // nal_hrd_parameters_present_flag
    w.write_bit(false); // vcl_hrd_parameters_present_flag
    w.write_bit(false); // pic_struct_present_flag
    w.write_bit(false); // bitstream_restriction_flag
    w.rbsp_trailing();
    nal(3, NAL_SPS, &w.finish())
}

/// Picture parameter set: CAVLC, the sequence's scaling, deblocking controlled from the slice headers,
/// the chroma QP offsets of [`CHROMA_QP_OFFSET`].
pub fn pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(0); // pic_parameter_set_id
    w.write_ue(0); // seq_parameter_set_id
    w.write_bit(false); // entropy_coding_mode_flag: CAVLC
    w.write_bit(false); // bottom_field_pic_order_in_frame_present_flag
    w.write_ue(0); // num_slice_groups_minus1
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.write_bit(false); // weighted_pred_flag
    w.write_bits(0, 2); // weighted_bipred_idc
    w.write_se(0); // pic_init_qp_minus26
    w.write_se(0); // pic_init_qs_minus26
    w.write_se(CHROMA_QP_OFFSET); // chroma_qp_index_offset
    w.write_bit(true); // deblocking_filter_control_present_flag
    w.write_bit(false); // constrained_intra_pred_flag
    w.write_bit(false); // redundant_pic_cnt_present_flag
    w.write_bit(true); // transform_8x8_mode_flag (the macroblocks say 4x4)
    w.write_bit(false); // pic_scaling_matrix_present_flag
    w.write_se(CHROMA_QP_OFFSET); // second_chroma_qp_index_offset
    w.rbsp_trailing();
    nal(3, NAL_PPS, &w.finish())
}

/// Access unit delimiter: I slices only (primary_pic_type 0).
pub fn aud() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 3);
    w.rbsp_trailing();
    nal(0, NAL_AUD, &w.finish())
}

/// Filler data NAL unit of exactly `bytes` bytes (at least 6: start code, header, 0x80).
pub fn filler(bytes: usize) -> Vec<u8> {
    let mut out = vec![0, 0, 0, 1, NAL_FILLER];
    out.resize(bytes.max(6) - 1, 0xFF);
    out.push(0x80);
    out
}

/// Slice header of an IDR I slice of an MBAFF frame (§7.3.3) starting at macroblock pair
/// `first_mb`, without deblocking: no decoder then differs from the encoder's reconstruction
/// (intra profiles do not require the filter, A.2.9).
pub fn slice_header(w: &mut BitWriter, first_mb: usize, idr_pic_id: u32, qp: i32) {
    w.write_ue(first_mb as u32);
    w.write_ue(7); // slice_type: I, all slices of the picture
    w.write_ue(0); // pic_parameter_set_id
    w.write_bits(0, 4); // frame_num (IDR)
    w.write_bit(false); // field_pic_flag: a frame
    w.write_ue(idr_pic_id);
    w.write_bits(0, 4); // pic_order_cnt_lsb
    // dec_ref_pic_marking of an IDR picture
    w.write_bit(false); // no_output_of_prior_pics_flag
    w.write_bit(false); // long_term_reference_flag
    w.write_se(qp - 26); // slice_qp_delta (pic_init_qp 26)
    w.write_ue(1); // disable_deblocking_filter_idc: off
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_and_filler() {
        assert_eq!(nal(0, 1, &[0, 0, 1, 0, 0, 0]), vec![0, 0, 0, 1, 1, 0, 0, 3, 1, 0, 0, 3, 0]);
        let f = filler(20);
        assert_eq!(f.len(), 20);
        assert_eq!(&f[..5], &[0, 0, 0, 1, 12]);
        assert_eq!(f[19], 0x80);
    }
}
