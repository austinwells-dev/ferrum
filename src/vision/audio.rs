//! Audio input for LFM3-Audio: WAV decoding, resampling to 16 kHz mono, and
//! the log-mel features its encoder reads.
//!
//! The features follow llama.cpp's `lfm3a` preprocessor: a periodic Hann
//! window of 400 samples, hop 160, no centring (the clip is zero-padded on the
//! right to a whole hop), a power spectrum through 128 Slaney mel filters with
//! Slaney area normalisation, `log10(max(x, 1e-12))`, then `(x + 4) / 4`.
#![forbid(unsafe_code)]

use crate::{Error, Result};

pub const SAMPLE_RATE: usize = 16_000;
const N_FFT: usize = 400;
const HOP: usize = 160;
const MEL_FLOOR: f64 = 1e-12;

/// A WAV file's format and sample bytes, read without decoding.
struct Wav<'a> {
    tag: u16,
    channels: usize,
    rate: usize,
    bits: usize,
    data: &'a [u8],
}

fn bad(what: &str) -> Error {
    Error::Parameter(format!("audio: {what}"))
}

fn parse_wav(bytes: &[u8]) -> Result<Wav<'_>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad(
            "only WAV files are read; convert other formats first (for example `afconvert -f WAVE -d LEI16@16000 in.m4a out.wav`)",
        ));
    }
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let mut format = None;
    let mut data = None;
    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = u32_at(bytes, at + 4) as usize;
        let body = &bytes[at + 8..(at + 8).saturating_add(len).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                let mut tag = u16_at(body, 0);
                // WAVE_FORMAT_EXTENSIBLE carries the real tag in its subformat GUID.
                if tag == 0xFFFE && body.len() >= 26 {
                    tag = u16_at(body, 24);
                }
                format = Some((
                    tag,
                    u16_at(body, 2) as usize,
                    u32_at(body, 4) as usize,
                    u16_at(body, 14) as usize,
                ));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        // Chunks are padded to an even length.
        at = at.saturating_add(8 + len + (len & 1));
    }
    let (tag, channels, rate, bits) = format.ok_or_else(|| bad("WAV file has no fmt chunk"))?;
    let data = data.ok_or_else(|| bad("WAV file has no data chunk"))?;
    if channels == 0 || rate == 0 {
        return Err(bad("WAV file declares no channels or no sample rate"));
    }
    let wav = Wav {
        tag,
        channels,
        rate,
        bits,
        data,
    };
    if !wav.bits.is_multiple_of(8) || wav.sample(&[0; 8]).is_none() {
        return Err(bad(&format!(
            "WAV sample format {tag} with {bits} bits is not read; use PCM or float"
        )));
    }
    Ok(wav)
}

impl Wav<'_> {
    fn sample(&self, b: &[u8]) -> Option<f32> {
        Some(match (self.tag, self.bits) {
            (1, 8) => (f32::from(b[0]) - 128.) / 128.,
            (1, 16) => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.,
            (1, 24) => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.,
            (1, 32) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.,
            (3, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            (3, 64) => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
            _ => return None,
        })
    }

    fn frames(&self) -> usize {
        self.data.len() / (self.bits / 8 * self.channels)
    }
}

/// Length in seconds of a WAV file, from its header alone, so a clip that is
/// too long can be refused before it is decoded and resampled.
pub fn wav_seconds(bytes: &[u8]) -> Result<f64> {
    let wav = parse_wav(bytes)?;
    Ok(wav.frames() as f64 / wav.rate as f64)
}

/// Decode a WAV file to mono samples at `SAMPLE_RATE`. Channels are
/// averaged. PCM (8, 16, 24 and 32 bit) and IEEE float (32 and 64 bit) are
/// read; anything else is refused with a hint to convert.
pub fn decode_wav(bytes: &[u8]) -> Result<Vec<f32>> {
    let wav = parse_wav(bytes)?;
    let width = wav.bits / 8;
    let mono: Vec<f32> = wav
        .data
        .chunks_exact(width * wav.channels)
        .map(|f| {
            let sum: f32 = f.chunks_exact(width).filter_map(|b| wav.sample(b)).sum();
            sum / wav.channels as f32
        })
        .collect();
    if mono.is_empty() {
        return Err(bad("WAV file has no samples"));
    }
    Ok(resample(&mono, wav.rate, SAMPLE_RATE))
}

/// Phase tables are built when the rates' ratio has at most this many
/// phases (every common rate against 16 kHz); otherwise each output's taps
/// are computed as it is made.
const MAX_PHASES: usize = 2048;

/// Windowed-sinc resampling (Blackman window, 32 zero crossings each side),
/// low-passed at the lower Nyquist rate. Output `i` sits at input position
/// `i * from / to`, whose fractional part cycles through `to / gcd` values,
/// so the kernel is tabulated once per phase rather than evaluated per tap.
pub fn resample(input: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    const ZEROS: f64 = 32.;
    let ratio = to as f64 / from as f64;
    // Cutoff relative to the input rate.
    let cutoff = ratio.min(1.) * 0.97;
    let half = ZEROS / cutoff;
    let len = ((input.len() as f64) * ratio).round() as usize;
    let g = gcd(from, to);
    let (up, down) = (to / g, from / g);
    // Taps for an output `frac` (in [0, 1)) past input sample `base`: offsets
    // `first..` from `base`, with their weights.
    let taps = |frac: f64| -> (i64, Vec<f64>) {
        let first = (frac - half).ceil() as i64;
        let last = (frac + half).floor() as i64;
        let weights = (first..=last)
            .map(|k| {
                let d = k as f64 - frac;
                let w = 0.42
                    + 0.5 * (std::f64::consts::PI * d / half).cos()
                    + 0.08 * (2. * std::f64::consts::PI * d / half).cos();
                cutoff * sinc(cutoff * d) * w
            })
            .collect();
        (first, weights)
    };
    let table: Vec<_> = if up <= MAX_PHASES {
        (0..up).map(|p| taps(p as f64 / up as f64)).collect()
    } else {
        Vec::new()
    };
    let n = input.len() as i64;
    (0..len)
        .map(|i| {
            let pos = i as u128 * down as u128;
            let (base, phase) = ((pos / up as u128) as i64, (pos % up as u128) as usize);
            let owned;
            let (first, weights) = if table.is_empty() {
                owned = taps(phase as f64 / up as f64);
                (owned.0, &owned.1)
            } else {
                (table[phase].0, &table[phase].1)
            };
            let start = base + first;
            let lo = (-start).max(0) as usize;
            let hi = ((n - start).max(0) as usize).min(weights.len());
            let mut acc = 0.;
            for k in lo..hi {
                acc += f64::from(input[(start + k as i64) as usize]) * weights[k];
            }
            acc as f32
        })
        .collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The mel filterbank and window for one bin count.
pub struct MelBank {
    n_mel: usize,
    /// `[n_mel, N_FFT / 2 + 1]`.
    filters: Vec<f32>,
    window: Vec<f32>,
    /// `e^{-2 pi i k / N_FFT}` for k in 0..N_FFT.
    twiddle: Vec<(f64, f64)>,
}

impl MelBank {
    pub fn new(n_mel: usize) -> Self {
        let bins = N_FFT / 2 + 1;
        let sr = SAMPLE_RATE as f64;
        // Slaney mel scale (librosa's default).
        let lin = 3. / 200.;
        let min_log_mel = 1000. * lin;
        let log_step = 6.4f64.ln() / 27.;
        let to_mel = |f: f64| {
            if f < 1000. {
                f * lin
            } else {
                min_log_mel + (f / 1000.).ln() / log_step
            }
        };
        let to_hz = |m: f64| {
            if m < min_log_mel {
                m / lin
            } else {
                1000. * ((m - min_log_mel) * log_step).exp()
            }
        };
        let hi = to_mel(sr / 2.);
        let hz: Vec<f64> = (0..n_mel + 2)
            .map(|i| to_hz(hi * i as f64 / (n_mel + 1) as f64))
            .collect();
        let step = sr / N_FFT as f64;
        let mut filters = vec![0f32; n_mel * bins];
        for m in 0..n_mel {
            let (l, c, r) = (hz[m], hz[m + 1], hz[m + 2]);
            let norm = 2. / (r - l).max(1e-30);
            for k in 0..bins {
                let f = k as f64 * step;
                let w = if f >= l && f <= c {
                    (f - l) / (c - l).max(1e-30)
                } else if f > c && f <= r {
                    (r - f) / (r - c).max(1e-30)
                } else {
                    0.
                };
                filters[m * bins + k] = (w * norm) as f32;
            }
        }
        let window = (0..N_FFT)
            .map(|i| {
                (0.5 * (1. - (2. * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())) as f32
            })
            .collect();
        let twiddle = (0..N_FFT)
            .map(|k| {
                let a = -2. * std::f64::consts::PI * k as f64 / N_FFT as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self {
            n_mel,
            filters,
            window,
            twiddle,
        }
    }

    /// Frames a clip of `samples` produces.
    pub fn frames(samples: usize) -> usize {
        padded_len(samples).saturating_sub(N_FFT) / HOP + 1
    }

    /// Log-mel features of 16 kHz mono `samples`, `[frames, n_mel]`.
    pub fn features(&self, samples: &[f32]) -> Vec<f32> {
        let mut padded = samples.to_vec();
        padded.resize(padded_len(samples.len()), 0.);
        let frames = Self::frames(samples.len());
        let bins = N_FFT / 2 + 1;
        let mut out = vec![0f32; frames * self.n_mel];
        let mut buf = vec![(0f64, 0f64); N_FFT];
        let mut power = vec![0f32; bins];
        for (t, row) in out.chunks_exact_mut(self.n_mel).enumerate() {
            let frame = &padded[t * HOP..t * HOP + N_FFT];
            for ((b, &x), &w) in buf.iter_mut().zip(frame).zip(&self.window) {
                *b = (f64::from(w * x), 0.);
            }
            let spectrum = self.fft(&buf, 1);
            for (p, &(re, im)) in power.iter_mut().zip(&spectrum) {
                *p = (re * re + im * im) as f32;
            }
            for (m, v) in row.iter_mut().enumerate() {
                let f = &self.filters[m * bins..(m + 1) * bins];
                let sum: f64 = power.iter().zip(f).map(|(&p, &w)| f64::from(p * w)).sum();
                *v = ((sum.max(MEL_FLOOR).log10() + 4.) / 4.) as f32;
            }
        }
        out
    }

    /// Recursive mixed-radix DFT of `x`, a sub-sequence of an `N_FFT` frame
    /// taken every `stride` samples: radix 2 while the length is even, a
    /// direct DFT on the odd remainder (25 for 400).
    fn fft(&self, x: &[(f64, f64)], stride: usize) -> Vec<(f64, f64)> {
        let n = x.len();
        let tw = |k: usize| self.twiddle[(k * stride) % N_FFT];
        if n % 2 == 1 {
            return (0..n)
                .map(|k| {
                    x.iter()
                        .enumerate()
                        .fold((0., 0.), |(re, im), (j, &(a, b))| {
                            let (c, s) = tw(j * k % n);
                            (re + a * c - b * s, im + a * s + b * c)
                        })
                })
                .collect();
        }
        let even: Vec<_> = x.iter().step_by(2).copied().collect();
        let odd: Vec<_> = x.iter().skip(1).step_by(2).copied().collect();
        let (e, o) = (self.fft(&even, stride * 2), self.fft(&odd, stride * 2));
        let mut out = vec![(0., 0.); n];
        for k in 0..n / 2 {
            let (c, s) = tw(k);
            let (a, b) = o[k];
            let t = (a * c - b * s, a * s + b * c);
            out[k] = (e[k].0 + t.0, e[k].1 + t.1);
            out[k + n / 2] = (e[k].0 - t.0, e[k].1 - t.1);
        }
        out
    }
}

/// Right-padded so the last frame is whole and no sample is dropped.
fn padded_len(n: usize) -> usize {
    if n <= N_FFT {
        N_FFT
    } else {
        n + (HOP - (n - N_FFT) % HOP) % HOP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        let align = channels * bits / 8;
        out.extend_from_slice(&(rate * u32::from(align)).to_le_bytes());
        out.extend_from_slice(&align.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn reads_pcm16_and_averages_channels() {
        let data: Vec<u8> = [16384i16, -16384, 8192, 8192]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let s = decode_wav(&wav(1, 2, 16000, 16, &data)).unwrap();
        assert_eq!(s, vec![0., 0.25]);
    }

    #[test]
    fn reads_float32_and_refuses_other_containers() {
        let data: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            decode_wav(&wav(3, 1, 16000, 32, &data)).unwrap(),
            vec![0.5, -0.25]
        );
        assert!(decode_wav(b"ID3\x04 not a wav file").is_err());
        assert!(decode_wav(&wav(2, 1, 16000, 4, &[0; 4])).is_err());
    }

    /// The resampler before its phase tables: every tap evaluated directly.
    fn resample_direct(input: &[f32], from: usize, to: usize) -> Vec<f32> {
        let ratio = to as f64 / from as f64;
        let cutoff = ratio.min(1.) * 0.97;
        let half = 32. / cutoff;
        let len = ((input.len() as f64) * ratio).round() as usize;
        (0..len)
            .map(|i| {
                let center = i as f64 / ratio;
                let lo = (center - half).ceil().max(0.) as usize;
                let hi = ((center + half).floor() as usize).min(input.len() - 1);
                let mut acc = 0.;
                for (j, &x) in input.iter().enumerate().take(hi + 1).skip(lo) {
                    let d = j as f64 - center;
                    let w = 0.42
                        + 0.5 * (std::f64::consts::PI * d / half).cos()
                        + 0.08 * (2. * std::f64::consts::PI * d / half).cos();
                    acc += f64::from(x) * cutoff * sinc(cutoff * d) * w;
                }
                acc as f32
            })
            .collect()
    }

    #[test]
    fn phase_tables_match_direct_resampling() {
        let input: Vec<f32> = (0..9000)
            .map(|i| ((i * 7919) % 2003) as f32 / 1001.5 - 1.)
            .collect();
        // Tabulated (one, 160 and 320 phases), and a ratio with too many
        // phases for a table.
        for from in [48_000, 44_100, 22_050, 8_000, 47_993] {
            let fast = resample(&input, from, SAMPLE_RATE);
            let slow = resample_direct(&input, from, SAMPLE_RATE);
            assert_eq!(fast.len(), slow.len(), "{from}");
            let err = fast
                .iter()
                .zip(&slow)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-5, "{from}: max difference {err}");
        }
    }

    #[test]
    fn reads_pcm24_extensible_and_skips_odd_chunks() {
        // 24-bit PCM in a WAVE_FORMAT_EXTENSIBLE fmt chunk, after a 3-byte
        // LIST chunk (padded to 4) that comes before fmt and data.
        let samples: [i32; 2] = [4_194_304, -8_388_608];
        let data: Vec<u8> = samples
            .iter()
            .flat_map(|v| v.to_le_bytes()[..3].to_vec())
            .collect();
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&0xFFFEu16.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&16000u32.to_le_bytes());
        fmt.extend_from_slice(&48000u32.to_le_bytes());
        fmt.extend_from_slice(&3u16.to_le_bytes());
        fmt.extend_from_slice(&24u16.to_le_bytes());
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&24u16.to_le_bytes());
        fmt.extend_from_slice(&4u32.to_le_bytes());
        // Subformat GUID: PCM (1) then the fixed tail.
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(b"abc\0");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        assert_eq!(decode_wav(&out).unwrap(), vec![0.5, -1.]);
        assert_eq!(wav_seconds(&out).unwrap(), 2. / 16000.);
    }

    #[test]
    fn duration_comes_from_the_header() {
        let data = vec![0u8; 48_000 * 2 * 4 * 3];
        // Three seconds of 48 kHz stereo float.
        assert_eq!(wav_seconds(&wav(3, 2, 48_000, 32, &data)).unwrap(), 3.);
        assert!(wav_seconds(b"not a wav").is_err());
    }

    #[test]
    fn resampling_keeps_a_tone() {
        let tone = |rate: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (2. * std::f32::consts::PI * 440. * i as f32 / rate as f32).sin())
                .collect()
        };
        let out = resample(&tone(48000, 48000), 48000, 16000);
        assert_eq!(out.len(), 16000);
        let want = tone(16000, 16000);
        // Away from the edges, where the kernel runs off the clip.
        let err = out[200..15800]
            .iter()
            .zip(&want[200..15800])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 2e-3, "max error {err}");
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let bank = MelBank::new(128);
        let x: Vec<(f64, f64)> = (0..N_FFT)
            .map(|i| (((i * 37) % 101) as f64 / 50. - 1., 0.))
            .collect();
        let fast = bank.fft(&x, 1);
        for k in [0, 1, 7, 100, 200] {
            let (mut re, mut im) = (0., 0.);
            for (j, &(a, _)) in x.iter().enumerate() {
                let ang = -2. * std::f64::consts::PI * (j * k) as f64 / N_FFT as f64;
                re += a * ang.cos();
                im += a * ang.sin();
            }
            assert!((fast[k].0 - re).abs() < 1e-9 && (fast[k].1 - im).abs() < 1e-9);
        }
    }

    #[test]
    fn frame_count_and_silence() {
        assert_eq!(MelBank::frames(1), 1);
        assert_eq!(MelBank::frames(400), 1);
        assert_eq!(MelBank::frames(401), 2);
        assert_eq!(MelBank::frames(16000), 99);
        let bank = MelBank::new(128);
        let f = bank.features(&[0.; 800]);
        assert_eq!(f.len(), MelBank::frames(800) * 128);
        // log10(1e-12) = -12, so silence is (-12 + 4) / 4.
        assert!(f.iter().all(|&v| (v + 2.).abs() < 1e-6));
    }

    #[test]
    fn a_tone_lands_in_its_mel_band() {
        let bank = MelBank::new(128);
        let s: Vec<f32> = (0..1600)
            .map(|i| (2. * std::f32::consts::PI * 1000. * i as f32 / 16000.).sin())
            .collect();
        let f = bank.features(&s);
        let row = &f[128 * 3..128 * 4];
        let peak = (0..128).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        // 1 kHz is mel 15 on the Slaney scale; band m is centred on
        // mel (m + 1) * hi / 129 with hi = mel(8 kHz) = 45.25.
        let centre = |m: usize| (m + 1) as f64 * 45.245 / 129.;
        assert!((centre(peak) - 15.).abs() < 0.5, "peak band {peak}");
    }
}
