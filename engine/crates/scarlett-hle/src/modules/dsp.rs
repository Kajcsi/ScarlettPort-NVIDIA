//! The console's audio DSP (`chad.dll`): streams that run blocks of float
//! samples through a coprocessor. A title connects once with a memory block
//! for the DSP, activates streams, and submits one command per block (gains,
//! an input and an output buffer, and for some kinds a third buffer).
//! Activation hands back a status block whose second word is the ticket of
//! the last completed command and whose fourth is its result; the title
//! checks it one block later and treats a late command as an overrun.
//! Commands complete when submitted here.
//!
//! Three kinds of stream:
//!
//! * 1, convolution with an impulse response (reverb), one or two channels.
//!   The response comes already in the frequency domain, in partitions of
//!   `frames` complex values each: a `2 * frames` point transform of
//!   `frames` samples and as many zeros, packed so that value 0 holds the DC
//!   bin as its real part and the Nyquist bin as its imaginary part (option
//!   8: one response per channel, one after the other). Overlap-save: the
//!   last two blocks are transformed, each partition multiplies the
//!   transform of the block that many blocks back, the sum is transformed
//!   back and its second half is the output. The command's gains scale the
//!   input and the output. A partition starting with the float bits
//!   0xFFFFACE1 holds metadata, not values.
//! * 2, forward transform of one mono block: `frames` samples, multiplied by
//!   the command's third buffer if there is one (a window), into `frames /
//!   2` values packed as above; option 4 gives magnitude and phase instead of
//!   real and imaginary parts (value 0 stays as it is).
//! * 3, the inverse: such values back to `frames` samples, scaled by
//!   1/`frames`, so kind 2 then kind 3 gives the block back.
//!
//! Until scar13 every output was silence. Voices a title sends only through
//! these (its own spectral effects, through 2 and 3) were not heard: on
//! GTA VI the other party in phone calls, and nothing at all under water
//! (scar12). The layouts and conventions are those omonkeyz/xbox-emulator
//! worked out (`crates/compat/chad`, GPL-3.0-or-later); this is our own
//! adaptation of them.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Mutex;

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_POINTER: HRESULT = 0x8000_4003_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

const CONVOLUTION: u32 = 1;
const FORWARD: u32 = 2;
const INVERSE: u32 = 3;
/// Option bits that need a stereo stream, and one convolution refuses.
const STEREO_OPTIONS: u32 = 2 | 8;
const NOT_FOR_CONVOLUTION: u32 = 4;
/// Transforms: magnitude and phase in place of real and imaginary parts.
const POLAR: u32 = 4;
/// Convolution: a response per channel.
const TWO_RESPONSES: u32 = 8;
/// The first float of a partition that holds metadata.
const METADATA: u32 = 0xFFFF_ACE1;

pub struct Dsp {
    max_streams: u32,
    active: Mutex<u32>,
}

/// What activation describes.
#[repr(C, packed)]
pub struct StreamDesc {
    kind: u32,
    options: u32,
    frames: u32,
    channels: u32,
    ir_floats: u32,
    ir: *const f32,
}

/// The status block a stream's owner polls.
#[repr(C)]
pub struct Status {
    submitted: u32,
    completed: u32,
    reserved: u32,
    result: HRESULT,
}

/// One block of work.
#[repr(C)]
pub struct Command {
    gains: [f32; 2],
    input: *const f32,
    output: *mut f32,
    /// Forward transforms: a block the input is multiplied by first (or null).
    multiplier: *const f32,
}

#[derive(Clone, Copy, Default, Debug, PartialEq)]
struct Bin {
    re: f32,
    im: f32,
}

/// A radix-2 transform of `n` points.
struct Fft {
    n: usize,
    reversed: Vec<u32>,
    twiddles: Vec<Bin>,
}

impl Fft {
    fn new(n: usize) -> Fft {
        let bits = n.trailing_zeros();
        let reversed = (0..n as u32).map(|i| i.reverse_bits() >> (32 - bits)).collect();
        let twiddles = (0..n / 2)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Bin { re: a.cos() as f32, im: a.sin() as f32 }
            })
            .collect();
        Fft { n, reversed, twiddles }
    }

    /// In place, unscaled either way.
    fn run(&self, d: &mut [Bin], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.reversed[i] as usize;
            if j > i {
                d.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let (half, step) = (len / 2, n / len);
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let mut w = self.twiddles[k * step];
                    if inverse {
                        w.im = -w.im;
                    }
                    let (a, b) = (d[start + k], d[start + k + half]);
                    let t = Bin { re: b.re * w.re - b.im * w.im, im: b.re * w.im + b.im * w.re };
                    d[start + k] = Bin { re: a.re + t.re, im: a.im + t.im };
                    d[start + k + half] = Bin { re: a.re - t.re, im: a.im - t.im };
                }
            }
            len <<= 1;
        }
    }

    /// `n` real samples to `n / 2` packed values (see the module's notes).
    fn forward(&self, time: &[f32], work: &mut [Bin], out: &mut [Bin]) {
        let (n, half) = (self.n, self.n / 2);
        for (w, &t) in work[..n].iter_mut().zip(time) {
            *w = Bin { re: t, im: 0.0 };
        }
        self.run(&mut work[..n], false);
        out[0] = Bin { re: work[0].re, im: work[half].re };
        out[1..half].copy_from_slice(&work[1..half]);
    }

    /// `n / 2` packed values back to `n` samples, in `work[..n]`'s real
    /// parts, scaled by 1/n.
    fn inverse(&self, packed: &[Bin], work: &mut [Bin]) {
        let (n, half) = (self.n, self.n / 2);
        work[0] = Bin { re: packed[0].re, im: 0.0 };
        work[half] = Bin { re: packed[0].im, im: 0.0 };
        for k in 1..half {
            work[k] = packed[k];
            work[n - k] = Bin { re: packed[k].re, im: -packed[k].im };
        }
        self.run(&mut work[..n], true);
        let scale = 1.0 / n as f32;
        for w in &mut work[..n] {
            w.re *= scale;
        }
    }
}

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

struct Convolution {
    /// Per response, per partition, `frames` packed values.
    responses: Vec<Vec<Vec<Bin>>>,
    /// Per channel: the last two blocks of (scaled) input.
    history: Vec<Vec<f32>>,
    /// Per channel, a ring of the last blocks' transforms, one per partition.
    spectra: Vec<Vec<Vec<Bin>>>,
    /// The ring's newest entry.
    at: usize,
}

impl Convolution {
    /// # Safety
    /// `ir` holds `responses * partitions * 2 * frames` floats.
    unsafe fn new(ir: *const f32, frames: usize, channels: usize, partitions: usize, responses: usize) -> Convolution {
        // SAFETY: the caller's.
        let floats = unsafe { std::slice::from_raw_parts(ir, responses * partitions * 2 * frames) };
        let responses: Vec<Vec<Vec<Bin>>> = floats
            .chunks_exact(partitions * 2 * frames)
            .map(|response| {
                let mut parts: Vec<Vec<Bin>> = response
                    .chunks_exact(2 * frames)
                    .map(|part| {
                        if part[0].to_bits() == METADATA {
                            vec![Bin::default(); frames]
                        } else {
                            part.chunks_exact(2).map(|c| Bin { re: finite(c[0]), im: finite(c[1]) }).collect()
                        }
                    })
                    .collect();
                // Silent tail partitions cost the same as the others.
                while parts.len() > 1 && parts.last().is_some_and(|p| p.iter().all(|b| *b == Bin::default())) {
                    parts.pop();
                }
                parts
            })
            .collect();
        let used = responses.iter().map(Vec::len).max().unwrap_or(1).max(1);
        Convolution {
            responses,
            history: vec![vec![0.0; 2 * frames]; channels],
            spectra: vec![vec![vec![Bin::default(); frames]; used]; channels],
            at: 0,
        }
    }

    /// One block of `channels` blocks of `frames` samples, one channel
    /// after the other.
    fn process(&mut self, fft: &Fft, work: &mut [Bin], sum: &mut [Bin], gains: [f32; 2], input: *const f32, output: *mut f32) {
        let frames = fft.n / 2;
        let ring = self.spectra.first().map_or(1, Vec::len);
        self.at = (self.at + 1) % ring;
        for c in 0..self.history.len() {
            let history = &mut self.history[c];
            history.copy_within(frames.., 0);
            for i in 0..frames {
                // SAFETY: the title's input holds one block of the stream.
                let v = if input.is_null() { 0.0 } else { unsafe { input.add(c * frames + i).read_unaligned() } };
                history[frames + i] = finite(v * gains[0]);
            }
            fft.forward(history, work, &mut self.spectra[c][self.at]);
            sum.fill(Bin::default());
            let response = &self.responses[c.min(self.responses.len() - 1)];
            for (k, h) in response.iter().enumerate() {
                let x = &self.spectra[c][(self.at + ring - k) % ring];
                sum[0].re += x[0].re * h[0].re;
                sum[0].im += x[0].im * h[0].im;
                for b in 1..frames {
                    sum[b].re += x[b].re * h[b].re - x[b].im * h[b].im;
                    sum[b].im += x[b].re * h[b].im + x[b].im * h[b].re;
                }
            }
            fft.inverse(sum, work);
            if !output.is_null() {
                for i in 0..frames {
                    // SAFETY: the title's output holds one block of the stream.
                    unsafe { output.add(c * frames + i).write_unaligned(finite(work[frames + i].re * gains[1])) };
                }
            }
        }
    }
}

enum Work {
    Convolution(Convolution),
    Forward { polar: bool },
    Inverse { polar: bool },
    /// A kind not known: silence, as before.
    Silent,
}

pub struct Stream {
    dsp: *mut Dsp,
    frames: u32,
    channels: u32,
    status: Box<Status>,
    work: Work,
    fft: Fft,
    scratch: Vec<Bin>,
    values: Vec<Bin>,
    samples: Vec<f32>,
}

impl Stream {
    fn run(&mut self, command: &Command) {
        let frames = self.frames as usize;
        let output = command.output;
        match &mut self.work {
            Work::Convolution(convolution) => {
                convolution.process(&self.fft, &mut self.scratch, &mut self.values, command.gains, command.input, output);
            }
            Work::Forward { polar } => {
                for i in 0..frames {
                    // SAFETY: the title's input (and multiplier) hold one block.
                    let mut v = if command.input.is_null() { 0.0 } else { unsafe { command.input.add(i).read_unaligned() } };
                    if !command.multiplier.is_null() {
                        // SAFETY: as above.
                        v *= unsafe { command.multiplier.add(i).read_unaligned() };
                    }
                    self.samples[i] = finite(v);
                }
                let values = &mut self.values[..frames / 2];
                self.fft.forward(&self.samples, &mut self.scratch, values);
                if *polar {
                    for b in &mut values[1..] {
                        let (re, im) = (f64::from(b.re), f64::from(b.im));
                        *b = Bin { re: re.hypot(im) as f32, im: im.atan2(re) as f32 };
                    }
                }
                if !output.is_null() {
                    for (k, b) in values.iter().enumerate() {
                        // SAFETY: the title's output holds `frames` floats.
                        unsafe {
                            output.add(2 * k).write_unaligned(finite(b.re));
                            output.add(2 * k + 1).write_unaligned(finite(b.im));
                        }
                    }
                }
            }
            Work::Inverse { polar } => {
                for k in 0..frames / 2 {
                    // SAFETY: the title's input holds `frames` floats.
                    let (re, im) = if command.input.is_null() {
                        (0.0, 0.0)
                    } else {
                        unsafe { (command.input.add(2 * k).read_unaligned(), command.input.add(2 * k + 1).read_unaligned()) }
                    };
                    self.values[k] = Bin { re: finite(re), im: finite(im) };
                }
                if *polar {
                    for b in &mut self.values[1..frames / 2] {
                        let (m, a) = (f64::from(b.re), f64::from(b.im));
                        *b = Bin { re: (m * a.cos()) as f32, im: (m * a.sin()) as f32 };
                    }
                }
                self.fft.inverse(&self.values, &mut self.scratch);
                if !output.is_null() {
                    for i in 0..frames {
                        // SAFETY: the title's output holds one block.
                        unsafe { output.add(i).write_unaligned(finite(self.scratch[i].re)) };
                    }
                }
            }
            Work::Silent => {
                if !output.is_null() {
                    let samples = frames * self.channels as usize;
                    // SAFETY: the title's output buffer holds one block of the stream.
                    unsafe { std::ptr::write_bytes(output, 0, samples) };
                }
            }
        }
    }
}

pub extern "system" fn XDspConnectWithMaximumStreamLimit(
    base: *mut c_void,
    base_length: u32,
    ir_seconds: u32,
    max_streams: u32,
    max_channels: u32,
    out: *mut *mut Dsp,
) -> HRESULT {
    if base.is_null()
        || !(0x1_0000..=0x1F40_0000).contains(&base_length)
        || ir_seconds > 0x80
        || max_streams > 0x100
        || max_channels > 2
    {
        return E_INVALIDARG;
    }
    if out.is_null() {
        return E_POINTER;
    }
    tracing::info!(
        base_length,
        ir_seconds,
        max_streams,
        max_channels,
        "audio DSP connected"
    );
    let dsp = Box::new(Dsp {
        max_streams,
        active: Mutex::new(0),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(dsp)) };
    S_OK
}

pub extern "system" fn XDspDisconnect(dsp: *mut Dsp) -> HRESULT {
    if dsp.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspConnectWithMaximumStreamLimit`; disconnected
    // once, after its streams.
    drop(unsafe { Box::from_raw(dsp) });
    S_OK
}

fn valid(desc: &StreamDesc) -> bool {
    let (kind, options, frames, channels) = (desc.kind, desc.options, desc.frames, desc.channels);
    let shape = if kind == CONVOLUTION {
        (1..=2).contains(&channels)
            && matches!(frames, 0x200 | 0x400)
            && options & NOT_FOR_CONVOLUTION == 0
    } else {
        channels == 1 && matches!(frames, 0x200 | 0x400 | 0x800)
    };
    shape && (options & STEREO_OPTIONS == 0 || channels == 2)
}

/// Says each kind of stream the title activates once.
fn first_of_its_kind(kind: u32, options: u32, frames: u32, channels: u32, partitions: usize) {
    static SEEN: Mutex<Option<HashSet<(u32, u32, u32, u32)>>> = Mutex::new(None);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.get_or_insert_with(HashSet::new).insert((kind, options, frames, channels)) {
        let what = match kind {
            CONVOLUTION => "convolution",
            FORWARD => "forward transform",
            INVERSE => "inverse transform",
            _ => "unknown kind, silent",
        };
        tracing::info!(kind, options, frames, channels, partitions, "audio DSP stream activated: {what}");
    }
}

pub extern "system" fn XDspActivate(
    dsp: *mut Dsp,
    desc: *const StreamDesc,
    status: *mut *const Status,
    stream: *mut *mut Stream,
) -> HRESULT {
    if status.is_null() || stream.is_null() {
        return E_POINTER;
    }
    if dsp.is_null() || desc.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes its description.
    let desc = unsafe { &*desc };
    if !valid(desc) {
        return E_INVALIDARG;
    }
    let (kind, options, frames, channels, ir_floats, ir) =
        (desc.kind, desc.options, desc.frames, desc.channels, desc.ir_floats, desc.ir);
    let n = frames as usize;
    let partitions = (ir_floats as usize / (2 * n)).max(1);
    {
        // SAFETY: a connection from `XDspConnectWithMaximumStreamLimit`.
        let dsp = unsafe { &*dsp };
        let mut active = dsp.active.lock().unwrap_or_else(|e| e.into_inner());
        if *active >= dsp.max_streams {
            return E_OUTOFMEMORY;
        }
        *active += 1;
    }
    let (work, points) = match kind {
        CONVOLUTION if !ir.is_null() => {
            let responses = if options & TWO_RESPONSES != 0 { 2 } else { 1 };
            // SAFETY: the description's response holds `ir_floats` floats per
            // response.
            let convolution = unsafe { Convolution::new(ir, n, channels as usize, partitions, responses) };
            (Work::Convolution(convolution), 2 * n)
        }
        FORWARD => (Work::Forward { polar: options & POLAR != 0 }, n),
        INVERSE => (Work::Inverse { polar: options & POLAR != 0 }, n),
        _ => (Work::Silent, 2),
    };
    first_of_its_kind(kind, options, frames, channels, partitions);
    let new = Box::new(Stream {
        dsp,
        frames,
        channels,
        status: Box::new(Status {
            submitted: 0,
            completed: 0,
            reserved: 0,
            result: S_OK,
        }),
        work,
        fft: Fft::new(points),
        scratch: vec![Bin::default(); points],
        values: vec![Bin::default(); n],
        samples: vec![0.0; n],
    });
    // SAFETY: checked non-null.
    unsafe {
        status.write(&*new.status);
        stream.write(Box::into_raw(new));
    }
    S_OK
}

pub extern "system" fn XDspDeactivate(stream: *mut Stream) -> HRESULT {
    if stream.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XDspActivate`; deactivated once.
    let stream = unsafe { Box::from_raw(stream) };
    // SAFETY: the connection outlives its streams.
    let dsp = unsafe { &*stream.dsp };
    *dsp.active.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
    S_OK
}

/// Runs the command and completes it at once.
pub extern "system" fn XDspSubmitCommand(
    stream: *mut Stream,
    command: *const Command,
    ticket: *mut u32,
) -> HRESULT {
    if stream.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a stream from `XDspActivate`, the title's command.
    let (stream, command) = unsafe { (&mut *stream, &*command) };
    stream.run(command);
    let status = &mut *stream.status;
    let next = status.submitted.wrapping_add(1);
    // SAFETY: the status block is read by title code on other threads.
    unsafe {
        std::ptr::write_volatile(&mut status.submitted, next);
        std::ptr::write_volatile(&mut status.result, S_OK);
        std::ptr::write_volatile(&mut status.completed, next);
    }
    if !ticket.is_null() {
        // SAFETY: optional out parameter.
        unsafe { ticket.write(next) };
    }
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dsp",
        dlls: &["chad.dll"],
        exports: vec![
            hle_export!(XDspConnectWithMaximumStreamLimit),
            hle_export!(XDspDisconnect),
            hle_export!(XDspActivate),
            hle_export!(XDspDeactivate),
            hle_export!(XDspSubmitCommand),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connect() -> *mut Dsp {
        let mut base = vec![0u8; 0x10_0000];
        let mut dsp = std::ptr::null_mut();
        assert_eq!(XDspConnectWithMaximumStreamLimit(base.as_mut_ptr().cast(), 0x10_0000, 24, 0x80, 1, &mut dsp), S_OK);
        dsp
    }

    fn activate(dsp: *mut Dsp, desc: &StreamDesc) -> (*const Status, *mut Stream) {
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        assert_eq!(XDspActivate(dsp, desc, &mut status, &mut stream), S_OK);
        (status, stream)
    }

    fn submit(stream: *mut Stream, input: &[f32], output: &mut [f32], multiplier: *const f32) -> u32 {
        let command = Command { gains: [1.0, 1.0], input: input.as_ptr(), output: output.as_mut_ptr(), multiplier };
        let mut ticket = 0;
        assert_eq!(XDspSubmitCommand(stream, &command, &mut ticket), S_OK);
        ticket
    }

    fn transform(kind: u32, options: u32, frames: u32) -> StreamDesc {
        StreamDesc { kind, options, frames, channels: 1, ir_floats: 0, ir: std::ptr::null() }
    }

    /// The plain DFT's bins 0..=n/2.
    fn dft(x: &[f32]) -> Vec<(f64, f64)> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                x.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, &v)| {
                    let a = -2.0 * std::f64::consts::PI * (k * t % n) as f64 / n as f64;
                    (re + f64::from(v) * a.cos(), im + f64::from(v) * a.sin())
                })
            })
            .collect()
    }

    #[test]
    fn forward_then_inverse_gives_the_block_back() {
        let dsp = connect();
        for frames in [0x200u32, 0x400, 0x800] {
            for options in [0, POLAR] {
                let n = frames as usize;
                let (status, forward) = activate(dsp, &transform(FORWARD, options, frames));
                let (_, inverse) = activate(dsp, &transform(INVERSE, options, frames));
                let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.31).sin() + 0.25 * ((i * 37 % 11) as f32 - 5.0) / 5.0).collect();
                let mut spectrum = vec![0f32; n];
                let ticket = submit(forward, &x, &mut spectrum, std::ptr::null());
                // SAFETY: the stream's status block.
                assert_eq!(unsafe { ((*status).completed, (*status).result) }, (ticket, S_OK));
                if options == 0 {
                    let want = dft(&x);
                    let tolerance = 2e-3 * n as f64;
                    assert!((f64::from(spectrum[0]) - want[0].0).abs() < tolerance);
                    assert!((f64::from(spectrum[1]) - want[n / 2].0).abs() < tolerance);
                    for k in 1..n / 2 {
                        assert!((f64::from(spectrum[2 * k]) - want[k].0).abs() < tolerance, "bin {k}");
                        assert!((f64::from(spectrum[2 * k + 1]) - want[k].1).abs() < tolerance, "bin {k}");
                    }
                }
                let mut back = vec![0f32; n];
                submit(inverse, &spectrum, &mut back, std::ptr::null());
                for i in 0..n {
                    assert!((back[i] - x[i]).abs() < 1e-3, "frames {frames} options {options} sample {i}");
                }
                assert_eq!(XDspDeactivate(forward), S_OK);
                assert_eq!(XDspDeactivate(inverse), S_OK);
            }
        }
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn the_forward_transform_takes_the_window_first() {
        let dsp = connect();
        let (_, forward) = activate(dsp, &transform(FORWARD, 0, 0x200));
        let x = vec![1f32; 0x200];
        let window = vec![0.5f32; 0x200];
        let mut spectrum = vec![0f32; 0x200];
        submit(forward, &x, &mut spectrum, window.as_ptr());
        // DC of 0x200 halves.
        assert!((spectrum[0] - 256.0).abs() < 1e-2 && spectrum[2..].iter().all(|v| v.abs() < 1e-2));
        assert_eq!(XDspDeactivate(forward), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn convolution_matches_the_direct_sum() {
        let dsp = connect();
        let (frames, partitions) = (0x200usize, 3usize);
        // A response of two and a half blocks, packed as the title packs it.
        let h: Vec<f32> = (0..frames * 5 / 2).map(|i| ((i * 7 % 13) as f32 - 6.0) / 40.0 * (-(i as f32) / 500.0).exp()).collect();
        let fft = Fft::new(2 * frames);
        let mut work = vec![Bin::default(); 2 * frames];
        let mut ir = Vec::new();
        for p in 0..partitions {
            let mut t = vec![0f32; 2 * frames];
            for i in 0..frames {
                t[i] = *h.get(p * frames + i).unwrap_or(&0.0);
            }
            let mut packed = vec![Bin::default(); frames];
            fft.forward(&t, &mut work, &mut packed);
            ir.extend(packed.iter().flat_map(|b| [b.re, b.im]));
        }
        let desc = StreamDesc {
            kind: CONVOLUTION,
            options: 0,
            frames: frames as u32,
            channels: 1,
            ir_floats: (partitions * 2 * frames) as u32,
            ir: ir.as_ptr(),
        };
        let (_, stream) = activate(dsp, &desc);
        let x: Vec<f32> = (0..frames * 4).map(|i| ((i as f32) * 0.07).sin() + if i % 97 == 0 { 1.0 } else { 0.0 }).collect();
        let mut y = vec![0f32; frames * 4];
        for b in 0..4 {
            submit(stream, &x[b * frames..(b + 1) * frames], &mut y[b * frames..(b + 1) * frames], std::ptr::null());
        }
        for (i, &got) in y.iter().enumerate() {
            let want: f32 = (0..=i).filter_map(|j| h.get(i - j).map(|&hv| hv * x[j])).sum();
            assert!((got - want).abs() < 1e-3, "sample {i}: {got} against {want}");
        }
        assert_eq!(XDspDeactivate(stream), S_OK);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }

    #[test]
    fn descriptions_the_console_refuses() {
        let dsp = connect();
        let ir = vec![0f32; 0x800];
        let stereo = StreamDesc { kind: CONVOLUTION, options: 2, frames: 0x400, channels: 2, ir_floats: 0x800, ir: ir.as_ptr() };
        let (_, stream) = activate(dsp, &stereo);
        let input = vec![0.5f32; 0x800];
        let mut output = vec![1f32; 0x800];
        submit(stream, &input, &mut output, std::ptr::null());
        // A silent response: silence.
        assert!(output.iter().all(|&s| s == 0.0));
        assert_eq!(XDspDeactivate(stream), S_OK);
        let (mut status, mut stream) = (std::ptr::null(), std::ptr::null_mut());
        // Mono streams refuse stereo options; transforms are mono only.
        let bad = StreamDesc { channels: 1, ..stereo };
        assert_eq!(XDspActivate(dsp, &bad, &mut status, &mut stream), E_INVALIDARG);
        let polar_convolution = StreamDesc { options: POLAR, ..stereo };
        assert_eq!(XDspActivate(dsp, &polar_convolution, &mut status, &mut stream), E_INVALIDARG);
        assert_eq!(XDspDisconnect(dsp), S_OK);
    }
}
