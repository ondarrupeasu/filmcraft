use super::*;

/// A moving test frame: a textured background panning right, a block moving down, noise; the
/// fields of each frame 20 ms apart (half the motion between them).
fn test_frame(n: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut s = n.wrapping_mul(2_654_435_761).max(1);
    let mut noise = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s % 9) as i32 - 4
    };
    let mut y = vec![0u8; WIDTH * HEIGHT];
    let mut cb = vec![0u8; WIDTH / 2 * HEIGHT];
    let mut cr = vec![0u8; WIDTH / 2 * HEIGHT];
    for r in 0..HEIGHT {
        let t = f64::from(n) + (r % 2) as f64 * 0.5;
        let pan = t * 6.0;
        for x in 0..WIDTH {
            let fx = x as f64 + pan;
            let fy = r as f64;
            let mut v = 40.0 + 150.0 * fx.rem_euclid(1920.0) / 1920.0 + 30.0 * (fx / 7.0).sin() * (fy / 11.0).cos();
            let (bx, by) = (900.0, 200.0 + t * 8.0);
            if (x as f64 - bx).abs() < 120.0 && (fy - by).abs() < 90.0 {
                v = 200.0 + 20.0 * ((x as f64 - bx) / 5.0).sin();
            }
            y[r * WIDTH + x] = (v as i32 + noise()).clamp(16, 235) as u8;
        }
        for x in 0..WIDTH / 2 {
            let fx = x as f64 * 2.0 + pan;
            cb[r * WIDTH / 2 + x] = (128.0 + 40.0 * (fx / 200.0).sin()) as u8;
            cr[r * WIDTH / 2 + x] = (128.0 + 30.0 * (r as f64 / 150.0).cos()) as u8;
        }
    }
    (y, cb, cr)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: f64 = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum();
    10.0 * (255.0f64 * 255.0 / (se / a.len() as f64).max(1e-9)).log10()
}

fn encode(frames: &[(Vec<u8>, Vec<u8>, Vec<u8>)]) -> Vec<Packet> {
    let mut enc = Encoder::new();
    let mut out = Vec::new();
    for f in frames {
        out.extend(enc.encode(&Frame { y: &f.0, cb: &f.1, cr: &f.2 }));
    }
    out.extend(enc.flush());
    out
}

/// FilmCraft's own MPEG-2 decoder reads back the long GOP in display order, near the source.
#[test]
fn our_decoder_reads_it_back() {
    let frames: Vec<_> = (0..26).map(test_frame).collect();
    let packets = encode(&frames);
    assert_eq!(packets.len(), frames.len());
    assert!(packets[0].key && packets[0].display == 0);
    // coding order: I0 P3 B1 B2 P6 B4 B5 …
    let order: Vec<u64> = packets.iter().map(|p| p.display).collect();
    assert_eq!(&order[..7], &[0, 3, 1, 2, 6, 4, 5]);
    assert!(packets.iter().filter(|p| p.key).map(|p| p.display).eq([0, 12, 24]));
    let total: usize = packets.iter().map(|p| p.data.len()).sum();
    let mbps = total as f64 * 8.0 * 25.0 / frames.len() as f64 / 1e6;
    eprintln!("{mbps:.2} Mbit/s; sizes {:?}", packets.iter().map(|p| p.data.len() / 1000).collect::<Vec<_>>());
    let mut dec = filmcraft_mpeg2v::Decoder::new();
    let mut pics = Vec::new();
    for p in &packets {
        pics.extend(dec.decode(&p.data, p.display as i64).expect("decodes"));
    }
    pics.extend(dec.flush());
    assert_eq!(dec.errors(), 0, "{:?}", dec.last_error());
    assert_eq!(pics.len(), frames.len());
    for (k, (pic, f)) in pics.iter().zip(&frames).enumerate() {
        assert_eq!((pic.width, pic.height), (1920, 1080));
        let (py, pc) = (psnr(&pic.y, &f.0), psnr(&pic.cb, &f.1));
        eprintln!("frame {k} {:?}: {py:.2} dB / {pc:.2} dB", pic.picture_type);
        assert!(py > 34.0 && pc > 34.0, "frame {k}: {py:.2} / {pc:.2}");
    }
    let stream: Vec<u8> = packets.iter().flat_map(|p| p.data.clone()).collect();
    if let Ok(keep) = std::env::var("M2V_KEEP") {
        std::fs::write(keep, &stream).expect("keep");
    }
    // ffmpeg's decoder (another IDCT) agrees with ours through the GOP: no drift
    if std::process::Command::new("ffmpeg").arg("-version").output().is_ok_and(|o| o.status.success()) {
        let dir = std::env::temp_dir().join(format!("m2v-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let (m2v, yuv) = (dir.join("a.m2v"), dir.join("a.yuv"));
        std::fs::write(&m2v, &stream).expect("write");
        let st = std::process::Command::new("ffmpeg").args(["-v", "error", "-y", "-i"]).arg(&m2v).args(["-f", "rawvideo", "-pix_fmt", "yuv422p"]).arg(&yuv).status().expect("ffmpeg");
        assert!(st.success());
        let raw = std::fs::read(&yuv).expect("decoded");
        let fsize = WIDTH * HEIGHT * 2;
        assert_eq!(raw.len(), fsize * pics.len());
        for (k, pic) in pics.iter().enumerate() {
            let ff = &raw[k * fsize..k * fsize + WIDTH * HEIGHT];
            let p = psnr(ff, &pic.y);
            assert!(p > 48.0, "frame {k}: ffmpeg and ours {p:.2} dB apart");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// A decoder's VBV buffer, filled at 50 Mbit/s from the first byte and emptied by each picture
/// at its decoding time (the first one's vbv_delay after its picture start code, then every
/// 40 ms), neither empties nor overflows; the rate settles at 50 Mbit/s.
#[test]
fn the_vbv_buffer_holds() {
    let frames: Vec<_> = (0..60).map(test_frame).collect();
    let packets = encode(&frames);
    let stream: Vec<u8> = packets.iter().flat_map(|p| p.data.clone()).collect();
    // picture start codes and their vbv_delay
    let mut pics = Vec::new();
    for i in 0..stream.len() - 8 {
        if stream[i..i + 4] == [0, 0, 1, 0] {
            let v = u32::from_be_bytes([0, stream[i + 5], stream[i + 6], stream[i + 7]]);
            pics.push((i, (v >> 3) & 0xFFFF));
        }
    }
    assert_eq!(pics.len(), frames.len());
    let r = BIT_RATE as f64;
    // the first picture start code's last byte arrives at (offset + 4) bytes; decoding after vbv_delay
    let t0 = (pics[0].0 + 4) as f64 * 8.0 / r + f64::from(pics[0].1) / 90_000.0;
    // each picture's bytes run to the next picture's headers (the sequence and GOP headers go with the I)
    let starts: Vec<usize> = packets.iter().scan(0usize, |a, p| {
        let s = *a;
        *a += p.data.len();
        Some(s)
    }).collect();
    for n in 0..packets.len() {
        let t = t0 + n as f64 / 25.0;
        let end = (starts[n] + packets[n].data.len()) as f64 * 8.0;
        let arrived = (r * t).min(stream.len() as f64 * 8.0);
        assert!(end <= arrived + 1.0, "picture {n} not all there at its decoding time: {end} > {arrived}");
        let held = arrived - starts[n] as f64 * 8.0;
        assert!(held <= VBV_BITS as f64 + 8.0 * 64.0, "picture {n}: the buffer holds {held} bits");
    }
    let mbps = stream.len() as f64 * 8.0 * 25.0 / frames.len() as f64 / 1e6;
    eprintln!("{mbps:.2} Mbit/s over {} frames", frames.len());
    assert!((mbps - 50.0).abs() < 6.0, "{mbps}");
}
