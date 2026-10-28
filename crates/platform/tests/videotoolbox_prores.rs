//! ProRes 422 through VideoToolbox (macOS, a Mac with a ProRes engine) against FilmCraft's ProRes
//! in Rust: every profile (Proxy, LT, 422, HQ), at a size that is not a multiple of the
//! macroblock, gives the same pictures within 1 in a sample (ProRes leaves its IDCT's rounding to
//! the decoder: Apple's, ffmpeg's and ours each differ by that; Proxy, coarsely quantized, by up to
//! 4 in a few samples), the same size, colour and pts,
//! also after a `reset` (the session stays: a seek is one sample). 4444 is left to the Rust
//! decoder. Skips on a Mac without a ProRes engine.
#![cfg(target_os = "macos")]

use filmcraft_frame::PixelData;
use filmcraft_isobmff::{FourCc, SampleEntry};
use filmcraft_prores::{ChromaFormat, Encoder, Frame, Profile};

const W: u32 = 642;
const H: u32 = 362;

/// A 10-bit 4:2:2 picture with gradients and detail (`k` moves it), Rec. 709.
fn picture(k: u32) -> Frame {
    let mut f = Frame::new(W, H, ChromaFormat::Yuv422, 10, false);
    let cw = ChromaFormat::Yuv422.chroma_width(W);
    for y in 0..H {
        for x in 0..W {
            let v = (64 + ((x * 3 + y * 2 + k * 17) % 876) as u16) ^ (((x / 3 + y / 5 + k) % 7) as u16 * 9);
            f.y[(y * W + x) as usize] = v.clamp(64, 940);
        }
        for x in 0..cw {
            f.cb[(y * cw + x) as usize] = (64 + (x * 5 + k * 31) % 896) as u16;
            f.cr[(y * cw + x) as usize] = (64 + (y * 7 + x + k * 13) % 896) as u16;
        }
    }
    f.color.matrix = 1;
    f.color.transfer = 1;
    f.color.primaries = 1;
    f
}

fn planes(d: &filmcraft_codecs::DecodedFrame) -> [Vec<u16>; 3] {
    match &d.frame.data {
        PixelData::Yuv16 { planes, bits: 10, .. } => [planes[0].to_vec(), planes[1].to_vec(), planes[2].to_vec()],
        other => panic!("expected 10-bit Y'CbCr, got {}", match other {
            PixelData::Yuv8 { .. } => "8-bit",
            _ => "RGBA",
        }),
    }
}

/// Largest difference between two planes.
fn worst(a: &[u16], b: &[u16]) -> u16 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0)
}

#[test]
fn prores_422_through_videotoolbox_matches_the_rust_decoder() {
    let profiles = [Profile::Proxy, Profile::Lt, Profile::Standard, Profile::Hq];
    for profile in profiles {
        let entry = SampleEntry::prores(FourCc(profile.fourcc()), W as u16, H as u16);
        let Some(hw) = filmcraft_platform::videotoolbox_factory(&entry) else {
            eprintln!("skipped: no ProRes engine on this Mac");
            return;
        };
        let mut hw = hw.expect("VideoToolbox ProRes");
        assert_eq!(hw.name(), "VideoToolbox ProRes");
        let mut sw = filmcraft_codecs::software_video_decoder(&entry).expect("Rust ProRes");
        let mut enc = Encoder::new(profile, W, H);
        let samples: Vec<Vec<u8>> = (0..4).map(|k| enc.encode(&picture(k)).expect("encodes")).collect();
        let mut got = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            got.extend(hw.decode(s, i as i64 * 1000).expect("hardware decodes"));
        }
        got.extend(hw.flush());
        assert_eq!(got.iter().map(|d| d.pts).collect::<Vec<_>>(), vec![0, 1000, 2000, 3000], "{profile:?}: every picture, in order");
        for (i, (d, s)) in got.iter().zip(&samples).enumerate() {
            let r = sw.decode(s, i as i64 * 1000).expect("software decodes").remove(0);
            assert_eq!((d.frame.width, d.frame.height), (r.frame.width, r.frame.height), "{profile:?}");
            assert_eq!(d.frame.color, r.frame.color, "{profile:?}: the colour as the Rust decoder reads it");
            let (a, b) = (planes(d), planes(&r));
            for p in 0..3 {
                if profile == Profile::Proxy {
                    // Proxy's coarse quantization makes the decoders' rounding tell more: a few
                    // samples (0.03 % of this busy picture) by up to 4.
                    let off = a[p].iter().zip(&b[p]).filter(|(x, y)| x.abs_diff(**y) > 1).count();
                    assert!(worst(&a[p], &b[p]) <= 4 && off * 1000 <= a[p].len(), "Proxy picture {i} plane {p}: off by {} in {off} samples", worst(&a[p], &b[p]));
                } else {
                    assert!(worst(&a[p], &b[p]) <= 1, "{profile:?} picture {i} plane {p}: off by {}", worst(&a[p], &b[p]));
                }
            }
        }
        // A seek: reset, then a picture alone, as it was.
        hw.reset();
        let again = hw.decode(&samples[2], 2000).expect("after reset");
        let again = if again.is_empty() { hw.flush() } else { again };
        assert_eq!(again.len(), 1);
        assert_eq!(planes(&again[0]), planes(&got[2]), "{profile:?}: the same picture after a reset");
    }
}

#[test]
fn prores_4444_is_left_to_the_rust_decoder() {
    let entry = SampleEntry::prores(FourCc(*b"ap4h"), W as u16, H as u16);
    assert!(filmcraft_platform::videotoolbox_factory(&entry).is_none());
}
