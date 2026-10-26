use super::*;

/// A busy 10-bit test frame: gradients, fine detail, hard edges and noise, the fields of it
/// apart (as if it moved).
fn test_frame(seed: u32) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    let mut noise = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 33) as i32 - 16
    };
    let mut y = vec![0u16; WIDTH * HEIGHT];
    let mut cb = vec![0u16; WIDTH / 2 * HEIGHT];
    let mut cr = vec![0u16; WIDTH / 2 * HEIGHT];
    for r in 0..HEIGHT {
        let shift = (r % 2) as f64 * 3.0 + f64::from(seed) * 7.0;
        for x in 0..WIDTH {
            let fx = x as f64 + shift;
            let fy = r as f64;
            let mut v = 64.0 + 700.0 * fx / WIDTH as f64 + 80.0 * (fx / 9.0).sin() * (fy / 13.0).cos();
            if (fx as usize / 160 + r / 120).is_multiple_of(2) && x > 900 {
                v += 120.0;
            }
            if x < 300 && r > 600 {
                v = 300.0 + 200.0 * ((fx * fx + fy * fy).sqrt() / 3.0).sin();
            }
            y[r * WIDTH + x] = (v as i32 + noise()).clamp(64, 940) as u16;
        }
        for x in 0..WIDTH / 2 {
            let fx = x as f64 * 2.0 + shift;
            cb[r * WIDTH / 2 + x] = (512.0 + 200.0 * (fx / 300.0).sin() + f64::from(noise()) * 0.5).clamp(64.0, 960.0) as u16;
            cr[r * WIDTH / 2 + x] = (512.0 + 150.0 * (r as f64 / 200.0).cos() - 100.0 * (fx > 1200.0) as i32 as f64).clamp(64.0, 960.0) as u16;
        }
    }
    (y, cb, cr)
}

#[test]
fn every_frame_is_its_fixed_size() {
    let mut enc = Encoder::new();
    for seed in 0..2 {
        let (y, cb, cr) = test_frame(seed);
        let out = enc.encode(&Frame { y: &y, cb: &cb, cr: &cr });
        assert_eq!(out.len(), FRAME_BYTES);
        assert_eq!(&out[..5], &[0, 0, 0, 1, 9], "an access unit delimiter first");
        assert_eq!(out[10] & 0x1F, 7, "then the SPS");
    }
}

/// ffmpeg's decoder (an independent one, when it is installed) reads the stream as 1080i 4:2:2
/// 10-bit and gives back the picture: PSNR over 45 dB.
#[test]
fn ffmpeg_decodes_it() {
    let Ok(ff) = std::process::Command::new("ffmpeg").arg("-version").output() else { return };
    if !ff.status.success() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("avci-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let mut enc = Encoder::new();
    let mut stream = Vec::new();
    let mut frames = Vec::new();
    for seed in 0..3 {
        let f = test_frame(seed);
        stream.extend(enc.encode(&Frame { y: &f.0, cb: &f.1, cr: &f.2 }));
        frames.push(f);
    }
    let h264 = dir.join("a.264");
    std::fs::write(&h264, &stream).expect("write");
    if let Ok(keep) = std::env::var("AVCI_KEEP") {
        std::fs::copy(&h264, keep).expect("keep");
    }
    let yuv = dir.join("a.yuv");
    let st = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "h264", "-i"])
        .arg(&h264)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv422p10le"])
        .arg(&yuv)
        .status()
        .expect("ffmpeg");
    assert!(st.success());
    let raw = std::fs::read(&yuv).expect("decoded");
    let fsize = (WIDTH * HEIGHT * 2) * 2;
    assert_eq!(raw.len(), fsize * frames.len(), "three 1920x1080 frames");
    for (k, (y, cb, _)) in frames.iter().enumerate() {
        let base = k * fsize;
        let sample = |i: usize| u16::from_le_bytes([raw[base + 2 * i], raw[base + 2 * i + 1]]);
        let mut se = 0f64;
        for i in 0..WIDTH * HEIGHT {
            let d = f64::from(sample(i)) - f64::from(y[i]);
            se += d * d;
        }
        let psnr = 10.0 * (1023.0f64 * 1023.0 / (se / (WIDTH * HEIGHT) as f64)).log10();
        let mut sc = 0f64;
        for i in 0..WIDTH / 2 * HEIGHT {
            let d = f64::from(sample(WIDTH * HEIGHT + i)) - f64::from(cb[i]);
            sc += d * d;
        }
        let psnr_c = 10.0 * (1023.0f64 * 1023.0 / (sc / (WIDTH / 2 * HEIGHT) as f64)).log10();
        eprintln!("frame {k}: luma {psnr:.2} dB, Cb {psnr_c:.2} dB");
        assert!(psnr > 45.0 && psnr_c > 45.0, "frame {k}: {psnr:.2} / {psnr_c:.2}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// How fast, at what QP, and how much of the frame is filler (`cargo test -- --ignored speed --nocapture`).
#[test]
#[ignore]
fn speed() {
    let mut enc = Encoder::new();
    let frames: Vec<_> = (0..6).map(test_frame).collect();
    let t = std::time::Instant::now();
    for f in &frames {
        let out = enc.encode(&Frame { y: &f.0, cb: &f.1, cr: &f.2 });
        // the filler is the last NAL unit: 0xFF bytes up to the end
        let fill = out.iter().rev().skip(1).take_while(|&&b| b == 0xFF).count();
        eprintln!("qp {} filler {fill} bytes", enc.qp);
    }
    eprintln!("{:.3} s a frame", t.elapsed().as_secs_f64() / frames.len() as f64);
}
