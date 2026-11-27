//! The title's buffer copies, done on the card at their place in the list
//! (`SCARLETT_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them (where its host copy
//! has one side only, `crate::shadow::card_all`).
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. So are those of more than 8 MiB out of
//! memory shaders do not write (streaming).

use super::*;
use crate::shadow::Shadow;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SCARLETT_COPIES_IN_PLACE").is_none_or(|v| v != "0"))
}

/// The most bytes copied out of memory shaders do not write.
const MOST: usize = 8 << 20;

/// Since the last frame-rate line: copies at their place, their bytes, those
/// out of a host copy, and texture copies out of one.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        let source = touches(written, self.src, self.src + self.len);
        if source && self.from.is_some() {
            return Waits::Yes;
        }
        match source || touches(written, self.dst, self.dst + self.len) {
            true => Waits::ForReaders,
            false => Waits::No,
        }
    }

    /// Whether the CPU takes the source's bytes at submission (the list
    /// copies out of its upload memory, not out of a host copy).
    pub(super) fn bytes_known(&self) -> bool {
        self.from.is_some()
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. The
    /// bytes, if they were taken, and whether title memory took them.
    pub(super) fn prepare(&self, free: bool) -> Option<(Vec<u8>, bool)> {
        if !free && self.from.is_none() {
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        if !free {
            return Some((bytes, false));
        }
        copy_mapped(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        // Title memory that is not all there was left alone.
        let mut there = vec![0u8; self.len];
        let took = read_into(self.dst as *const u8, &mut there) && there == bytes;
        Some((bytes, took))
    }
}

/// Whether a fill of upload memory that reads only what copies at their
/// place earlier in the submission wrote takes their bytes as the CPU took
/// them at submission, instead of waiting for flights
/// (`SCARLETT_PLACED_BYTES=0`: it waits, as before).
///
/// Such a copy writes the destination over whatever flights in the air put
/// there, so title memory, which flights may still write, is not read: but
/// where a flight may write the source or the destination, title memory does
/// not take the copy's bytes at submission ([`Placed::prepare`]), and every
/// fill behind it that read them waited for the flights, and was not done on
/// the card either ([`card_steps`]: it reads what a copy at its place
/// writes). Run scar13: some 100 ms a frame, one wait a frame, for a fill
/// of 560 bytes of constants.
pub(super) fn serve_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SCARLETT_PLACED_BYTES").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: fills that took a copy's bytes so.
pub(super) static SERVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Title memory, at a step of a submission, that copies at their place
/// earlier in it wrote with bytes the CPU takes at submission, out of a
/// source no flight in the air writes ([`serve_fills`]). In order, joined.
#[derive(Default)]
pub(super) struct Known(Vec<(usize, usize)>);

impl Known {
    /// A step writes `[from, to)`: with bytes known (`known`) or not.
    fn write(&mut self, from: usize, to: usize, known: bool) {
        if from >= to || (self.0.is_empty() && !known) {
            return;
        }
        let mut out = Vec::with_capacity(self.0.len() + 2);
        for &(a, b) in &self.0 {
            if b <= from || to <= a {
                out.push((a, b));
                continue;
            }
            if a < from {
                out.push((a, from));
            }
            if to < b {
                out.push((to, b));
            }
        }
        if known {
            out.push((from, to));
        }
        merge_ranges(&mut out);
        self.0 = out;
    }

    /// Whether `[from, to)` is all known.
    fn covers(&self, from: usize, to: usize) -> bool {
        let at = self.0.partition_point(|r| r.1 <= from);
        self.0.get(at).is_some_and(|r| r.0 <= from && to <= r.1)
    }

    /// Follows `step` against `written`, what flights may write: whether it
    /// is a fill that reads known bytes only.
    pub(super) fn serves(&mut self, step: &Step, written: &[(usize, usize)]) -> bool {
        if !serve_fills() {
            return false;
        }
        match step {
            Step::Placed(p) => {
                let known = p.bytes_known() && !touches(written, p.src, p.src + p.len);
                self.write(p.dst, p.dst + p.len, known);
                false
            }
            &Step::Copy(dst, _, len) => {
                self.write(dst, dst + len, false);
                false
            }
            Step::Run(_) => {
                self.0.clear();
                false
            }
            &Step::Fill(_, src, len, _) => !self.0.is_empty() && self.covers(src, src + len),
            _ => false,
        }
    }
}

/// The bytes of [`Known`], at a step of a submission being prepared: the
/// title address of each, not overlapping.
#[derive(Default)]
pub(super) struct Overlay(Vec<(usize, Vec<u8>)>);

impl Overlay {
    /// A step wrote `len` bytes at `at`: these (`Some`), or ones title
    /// memory holds now.
    pub(super) fn write(&mut self, at: usize, len: usize, bytes: Option<Vec<u8>>) {
        let to = at + len;
        if len == 0 || (self.0.is_empty() && bytes.is_none()) {
            return;
        }
        let mut out = Vec::with_capacity(self.0.len() + 2);
        for (a, held) in std::mem::take(&mut self.0) {
            let b = a + held.len();
            if b <= at || to <= a {
                out.push((a, held));
                continue;
            }
            if to < b {
                out.push((to, held[to - a..].to_vec()));
            }
            if a < at {
                let mut held = held;
                held.truncate(at - a);
                out.push((a, held));
            }
        }
        if let Some(bytes) = bytes {
            out.push((at, bytes));
        }
        self.0 = out;
    }

    /// Something wrote title memory anywhere.
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }

    /// Fills `len` bytes at `dst` with title memory at `src` as the steps so
    /// far left it, if bytes held here lie in it: `false` if none do.
    pub(super) fn fill(&self, dst: usize, src: usize, len: usize) -> bool {
        let to = src + len;
        let mut met = false;
        let mut covered = 0;
        for (a, held) in &self.0 {
            let b = a + held.len();
            if *a < to && src < b {
                met = true;
                covered += b.min(to) - (*a).max(src);
            }
        }
        if !met {
            return false;
        }
        if covered < len {
            copy_mapped(dst as *mut u8, src as *const u8, len, false);
        }
        for (a, held) in &self.0 {
            let b = a + held.len();
            if *a < to && src < b {
                let (from, until) = ((*a).max(src), b.min(to));
                // SAFETY: the fill's own upload memory of `len` bytes; the
                // part of it that lies over these.
                unsafe {
                    std::ptr::copy_nonoverlapping(held.as_ptr().add(from - a), (dst + (from - src)) as *mut u8, until - from);
                }
            }
        }
        true
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Whether a step of `runs` reads one of `ranges` on the CPU. `served`: the
/// fills (their list by address, their place in it, in order) that read
/// known bytes instead ([`Known`]).
pub(super) fn read_by(runs: &[&HostList], mut ranges: Vec<(usize, usize)>, served: &[(usize, usize)]) -> bool {
    join_ranges(&mut ranges, []);
    runs.iter().any(|list| {
        let at = std::ptr::from_ref::<HostList>(list) as usize;
        list.deferred().copies.iter().enumerate().any(|(i, step)| match step {
            Step::Fill(..) if served.binary_search(&(at, i)).is_ok() => false,
            &Step::Copy(_, src, len) | &Step::Fill(_, src, len, _) => touches(&ranges, src, src + len),
            Step::Texture(staging) | Step::TextureReady(staging) => {
                let (at, len) = staging.memory();
                touches(&ranges, at, at + len)
            }
            &Step::Plane((at, len), _) => touches(&ranges, at, at.saturating_add(len)),
            Step::Host(_, (from, to)) => touches(&ranges, *from, *to),
            Step::Placed(p) => p.from.is_some() && touches(&ranges, p.src, p.src + p.len),
            _ => false,
        })
    })
}

/// Everything recorded so far is done, and seen, before what follows.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all(host: &Host, list: &ID3D12GraphicsCommandList) {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    let enhanced = *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    });
    // SAFETY: the caller's.
    unsafe {
        if enhanced && let Ok(list) = list.cast::<ID3D12GraphicsCommandList7>() {
            let all = [D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            }];
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: all.as_ptr(),
                },
            }]);
            return;
        }
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        list.ResourceBarrier(&[alias, uav]);
    }
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
        let scratch = match &own {
            Some(_) => None,
            None if len <= MOST => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
            None => return false,
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        // SAFETY (both): an open list; live buffers (the list holds the
        // copies and its scratch memory), ranges inside them.
        unsafe { wait_for_all(host, &self.list) };
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if from.buffer.as_raw() == to.buffer.as_raw() {
                    let Some(between) = self.between(host, len as u64) else {
                        return false;
                    };
                    unsafe {
                        self.list.CopyBufferRegion(&between, 0, &from.buffer, offset, len as u64);
                        self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                        wait_for_all(host, &self.list);
                        self.list.CopyBufferRegion(&to.buffer, at, &between, 0, len as u64);
                    }
                } else {
                    unsafe { self.list.CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64) };
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None) => return false,
        }
        unsafe { wait_for_all(host, &self.list) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            from: scratch.map(|s| s.2 as usize),
        }));
        deferred.title_written();
        self.changed_something();
        true
    }

    /// For a copy of `[src, src + len)` into a texture: a buffer (readable by
    /// copies) that holds what shaders wrote there, at this place of the list.
    /// `None` where shaders do not write it.
    pub(crate) fn place_source(&self, src: usize, len: usize) -> Option<ID3D12Resource> {
        if !on() {
            return None;
        }
        let host = host()?;
        let from = shaders_own(src, len)?;
        let between = self.between(host, len as u64)?;
        from.refresh(self, src as u64, len as u64);
        STATS[3].fetch_add(1, Ordering::Relaxed);
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(&between, 0, &from.buffer, from.offset(src as u64), len as u64);
            self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            wait_for_all(host, &self.list);
        }
        Some(between)
    }

    /// A buffer on the card for this recording alone.
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        let buffer = host.uav_buffer(len.next_multiple_of(1 << 16))?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        Some(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::{Known, Overlay};

    #[test]
    fn known_bytes_follow_the_writes() {
        let mut k = Known::default();
        k.write(100, 200, true);
        assert!(k.covers(120, 180) && k.covers(100, 200));
        assert!(!k.covers(90, 120) && !k.covers(150, 210));
        // Written over with bytes not known: the rest stays known.
        k.write(150, 160, false);
        assert!(!k.covers(120, 180));
        assert!(k.covers(100, 150) && k.covers(160, 200));
        // Known again, and joined with what touches it.
        k.write(150, 160, true);
        assert!(k.covers(100, 200));
        k.write(200, 220, true);
        assert!(k.covers(180, 220));
        k.0.clear();
        assert!(!k.covers(100, 101));
    }

    #[test]
    fn fills_take_the_bytes_held() {
        let title = vec![1u8; 64];
        let src = title.as_ptr() as usize;
        let mut o = Overlay::default();
        let mut dst = vec![0u8; 64];
        // Nothing held: the fill is left to title memory.
        assert!(!o.fill(dst.as_mut_ptr() as usize, src, 64));
        o.write(src + 16, 16, Some(vec![9; 16]));
        assert!(o.fill(dst.as_mut_ptr() as usize, src, 64));
        assert_eq!(&dst[..16], &[1; 16]);
        assert_eq!(&dst[16..32], &[9; 16]);
        assert_eq!(&dst[32..], &[1; 32]);
        // Partly written over by a step that left its bytes in title memory.
        o.write(src + 20, 4, None);
        let mut dst = vec![0u8; 16];
        assert!(o.fill(dst.as_mut_ptr() as usize, src + 16, 16));
        assert_eq!(dst, [9, 9, 9, 9, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9]);
        // All of it held: title memory is not read.
        let mut dst = vec![0u8; 4];
        assert!(o.fill(dst.as_mut_ptr() as usize, src + 24, 4));
        assert_eq!(dst, [9; 4]);
        // Newer bytes over older ones.
        o.write(src + 26, 4, Some(vec![5; 4]));
        let mut dst = vec![0u8; 8];
        assert!(o.fill(dst.as_mut_ptr() as usize, src + 24, 8));
        assert_eq!(dst, [9, 9, 5, 5, 5, 5, 9, 9]);
        o.clear();
        assert!(!o.fill(dst.as_mut_ptr() as usize, src, 8));
        drop(title);
    }

    /// What copies with two sides cost to make (ignored: a measurement for
    /// `SCARLETT_CARD_ALL`): buffers of 64 KiB on the card and in the PC's
    /// memory.
    #[test]
    #[ignore]
    fn small_buffers_cost() {
        let Some(host) = super::host() else {
            return;
        };
        const N: usize = 1000;
        let used = || host.video_memory().map_or(0, |m| m[1] >> 20);
        let before = used();
        let t = std::time::Instant::now();
        let card: Vec<_> = (0..N).map(|_| host.uav_buffer(1 << 16)).collect();
        let card_us = t.elapsed().as_micros() as f64 / N as f64;
        let after = used();
        let t = std::time::Instant::now();
        let pc: Vec<_> = (0..N).map(|_| host.shared_buffer(1 << 16)).collect();
        let pc_us = t.elapsed().as_micros() as f64 / N as f64;
        println!(
            "{N} buffers of 64 KiB: on the card {card_us:.0} us each, {} MB of its memory; in the PC's memory {pc_us:.0} us each",
            after.saturating_sub(before)
        );
        drop((card, pc));
    }
}
