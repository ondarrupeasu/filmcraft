//! The broadcast HD codecs of MXF exports: AVC-Intra Class 100 and XDCAM HD422 (MPEG-2 4:2:2
//! long GOP), both 1920x1080 interlaced at 25 frames per second, top field first. The frames
//! come progressive: both fields of a frame are of the same instant (as 25p material is
//! delivered as 1080i25).

use filmcraft_isobmff::{FourCc, SampleEntry};
use filmcraft_time::FrameRate;

use crate::{EncodedPacket, EncoderFrame, ExportError, ExportSettings, MxfVideoCodec, Result, VideoEncoder, rgba_to_yuv422_10, rgbf_to_yuv422_10};

/// The encoder of `settings.mxf_video_codec`, when it is one of these.
pub(crate) fn encoder(settings: &ExportSettings, w: u32, h: u32, rate: FrameRate) -> Option<Result<Box<dyn VideoEncoder>>> {
    let codec = settings.mxf_video_codec;
    if !settings.format.is_mxf() || !codec.is_broadcast() {
        return None;
    }
    Some(check(w, h, rate).map(|()| -> Box<dyn VideoEncoder> {
        match codec {
            MxfVideoCodec::AvcIntra100 => Box::new(AvcIntra { enc: filmcraft_avcintra::Encoder::new(), rate, signal: settings.signal, buf: Yuv::default() }),
            _ => Box::new(Xdcam { enc: filmcraft_mpeg2enc::Encoder::new(), rate, signal: settings.signal, buf: Yuv::default(), coded: 0 }),
        }
    }))
}

/// These codecs are 1920x1080 at 25 frames per second.
pub(crate) fn check(w: u32, h: u32, rate: FrameRate) -> Result<()> {
    if (w, h) != (1920, 1080) || rate.num != rate.den * 25 {
        return Err(ExportError::Unsupported(format!("AVC-Intra 100 and XDCAM HD422 are 1920x1080 at 25 fps (not {w}x{h} at {})", rate.label())));
    }
    Ok(())
}

#[derive(Default)]
struct Yuv {
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,
}

impl Yuv {
    /// 10-bit BT.709 video range 4:2:2 of the frame.
    fn fill(&mut self, f: &EncoderFrame, signal: crate::ColorSignal) {
        let (w, h) = (f.width as usize, f.height as usize);
        self.y.resize(w * h, 0);
        self.cb.resize(w / 2 * h, 0);
        self.cr.resize(w / 2 * h, 0);
        match f.hdr {
            Some(rgb) => {
                let (kr, kb) = signal.kr_kb();
                rgbf_to_yuv422_10(rgb, w, h, kr, kb, &mut self.y, &mut self.cb, &mut self.cr)
            }
            None => rgba_to_yuv422_10(f.rgba, w, h, &mut self.y, &mut self.cb, &mut self.cr),
        }
    }
}

fn entry() -> SampleEntry {
    // MXF only: no MPEG-4 sample entry is written
    SampleEntry::dnx(FourCc(*b"mxfv"), 1920, 1080)
}

struct AvcIntra {
    enc: filmcraft_avcintra::Encoder,
    rate: FrameRate,
    signal: crate::ColorSignal,
    buf: Yuv,
}

impl VideoEncoder for AvcIntra {
    fn sample_entry(&self) -> SampleEntry {
        entry()
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        self.buf.fill(f, self.signal);
        let data = self.enc.encode(&filmcraft_avcintra::Frame { y: &self.buf.y, cb: &self.buf.cb, cr: &self.buf.cr });
        Ok(vec![EncodedPacket { data, key: true, duration: self.rate.den as u32, composition_offset: 0 }])
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        Ok(Vec::new())
    }
}

struct Xdcam {
    enc: filmcraft_mpeg2enc::Encoder,
    rate: FrameRate,
    signal: crate::ColorSignal,
    buf: Yuv,
    /// Pictures handed out so far (coding order).
    coded: i64,
}

impl Xdcam {
    /// Coding-order packets with their display position as the composition offset: one frame
    /// period more than (display − coded), never negative (a B picture shows a frame before it
    /// is coded at most).
    fn packets(&mut self, ps: Vec<filmcraft_mpeg2enc::Packet>) -> Vec<EncodedPacket> {
        let den = self.rate.den;
        ps.into_iter()
            .map(|p| {
                let off = (p.display as i64 - self.coded + 1) * den;
                self.coded += 1;
                EncodedPacket { data: p.data, key: p.key, duration: den as u32, composition_offset: off as i32 }
            })
            .collect()
    }
}

impl VideoEncoder for Xdcam {
    fn sample_entry(&self) -> SampleEntry {
        entry()
    }
    fn timescale(&self) -> u32 {
        self.rate.num as u32
    }
    fn encode(&mut self, f: &EncoderFrame) -> Result<Vec<EncodedPacket>> {
        self.buf.fill(f, self.signal);
        // 8-bit for MPEG-2
        let eight = |v: &[u16]| -> Vec<u8> { v.iter().map(|&s| ((s + 2) >> 2).min(255) as u8).collect() };
        let (y, cb, cr) = (eight(&self.buf.y), eight(&self.buf.cb), eight(&self.buf.cr));
        let ps = self.enc.encode(&filmcraft_mpeg2enc::Frame { y: &y, cb: &cb, cr: &cr });
        Ok(self.packets(ps))
    }
    fn flush(&mut self) -> Result<Vec<EncodedPacket>> {
        let ps = self.enc.flush();
        Ok(self.packets(ps))
    }
    fn media_start(&self) -> Option<i64> {
        Some(self.rate.den)
    }
}
