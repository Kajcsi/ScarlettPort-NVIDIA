//! Memory `allocate_upload` (`list.rs`) hands out: the title's CPU writes
//! it and its shaders read it at the same address (the structure builder's
//! instance descriptions lie there).
//!
//! The console's GPU addresses have 40 bits, and a translated shader that
//! reads by address has no more (`scarlett_shader`'s `Lower::memory`): memory
//! from the process heap lies above 2^40 and was read at another address, as
//! zeros. So this memory is taken in blocks of 64 KiB out of arenas in
//! [4 GiB, 2^37), as upstream ScarlettPort 0f637570 and 6d021fe3 place it (there
//! an allocation of its own each). An arena is a title buffer to the host
//! copies ([`crate::shadow`]) and lives as long as the process: the directory
//! of memory by address ([`crate::memory`]) keeps the copy it found at an
//! address for good, so the same addresses must stay the same copy.
//! `SCARLETT_UPLOAD_LOW=0`: from the process heap, as before.

use std::sync::{Mutex, OnceLock};

/// What blocks are cut on: a page of the directory of memory by address.
const BLOCK: usize = 64 << 10;
/// An arena's size, if one request does not need more.
const ARENA: usize = 4 << 20;

struct Arena {
    at: usize,
    /// Per block whether it is handed out.
    taken: Vec<bool>,
}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());

/// Upload memory of one request, zeroed, given back when dropped.
pub(crate) struct Upload(Kind);

enum Kind {
    /// Address, bytes.
    Low(usize, usize),
    Heap(Vec<u8>),
}

fn low() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SCARLETT_UPLOAD_LOW").is_none_or(|v| v != "0"))
}

/// The first place of `n` free blocks side by side.
fn gap(taken: &[bool], n: usize) -> Option<usize> {
    let mut free = 0;
    for (i, &t) in taken.iter().enumerate() {
        free = if t { 0 } else { free + 1 };
        if free == n {
            return Some(i + 1 - n);
        }
    }
    None
}

/// Address space every [`reserve`] is cut from first, taken before the
/// title runs ([`reserve_window`]). The title's own memory is placed from
/// 4 GiB up too, and what was reserved only when first needed could find
/// no place left: at 2560x1440 on the Series S profile a list's upload
/// memory asked for at the first frame in the world was refused and the
/// title's BVH builder threw "Failed to allocate upload buffer" (scar11).
/// Everything cut from it lives as long as the process, as before.
const WINDOW: usize = 2 << 30;
/// Committed with the window, so the first uploads never wait on commit.
const WINDOW_COMMITTED: usize = 64 << 20;

/// The window's start (0: none) and the bytes cut from it.
static WINDOW_AT: Mutex<(usize, usize)> = Mutex::new((0, 0));

/// Reserves the [`WINDOW`]; called once before the title runs. Without it
/// (or when it is used up) each [`reserve`] takes a place of its own.
pub fn reserve_window() {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
    if !low() {
        return;
    }
    let mut window = WINDOW_AT.lock().unwrap_or_else(|e| e.into_inner());
    if window.0 != 0 {
        return;
    }
    let Some(at) = place(WINDOW, false) else {
        explain_refusal(WINDOW, "address space for lists' uploads not reserved at start");
        return;
    };
    // SAFETY: the start of the reservation just made.
    let committed = !unsafe { VirtualAlloc(at as *const _, WINDOW_COMMITTED, MEM_COMMIT, PAGE_READWRITE) }.is_null();
    *window = (at, 0);
    tracing::info!(
        at = format_args!("{at:#x}"),
        bytes = WINDOW,
        committed = if committed { WINDOW_COMMITTED } else { 0 },
        "address space for lists' uploads and the directory of memory by address, reserved at start"
    );
}

/// Says once in a while why a reservation was refused: Windows' error (1455:
/// the commit limit; 8: no place), the free commit, and the free address
/// space where these must lie. Call it right after the refused call.
fn explain_refusal(len: usize, what: &str) {
    use std::sync::atomic::{AtomicU32, Ordering};
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: no arguments.
    let error = unsafe { GetLastError() };
    static SAID: AtomicU32 = AtomicU32::new(0);
    if SAID.fetch_add(1, Ordering::Relaxed) >= 8 {
        return;
    }
    let (mut free, mut largest) = (0usize, 0usize);
    let mut at = 1usize << 32;
    while at < 1 << 37 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let end = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(1 << 37);
        if end <= at {
            break;
        }
        if info.State == MEM_FREE {
            free += end - at;
            largest = largest.max(end - at);
        }
        at = end;
    }
    // SAFETY: the API fills in a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: as above.
    let commit = if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullAvailPageFile >> 20 } else { 0 };
    tracing::warn!(
        bytes = len,
        error,
        available_commit_mib = commit,
        free_below_2_37_mib = free >> 20,
        largest_free_below_2_37_mib = largest >> 20,
        "{what}"
    );
}

/// Fresh zeroed memory in [4 GiB, 2^37) (where the directory of memory by
/// address reaches, `MEMORY_PAGES`), write-watched as the title's own
/// pools are (copies then take only written pages): cut from the
/// [`WINDOW`] while it has room, else a reservation of its own.
pub(crate) fn reserve(len: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
    let cut = len.max(1).next_multiple_of(BLOCK);
    {
        let mut window = WINDOW_AT.lock().unwrap_or_else(|e| e.into_inner());
        let (start, used) = *window;
        if start != 0 && used + cut <= WINDOW {
            let at = start + used;
            // SAFETY: inside the window's reservation, never handed out
            // before (so still zeroed).
            if !unsafe { VirtualAlloc(at as *const _, cut, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
                window.1 += cut;
                return Some(at);
            }
            explain_refusal(cut, "memory for lists' uploads refused (commit in the window reserved at start)");
        }
    }
    let at = place(len, true);
    if at.is_none() {
        explain_refusal(len, "memory for lists' uploads refused");
    }
    at
}

/// A fresh reservation in [4 GiB, 2^37), write-watched, committed or not.
fn place(len: usize, commit: bool) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE, MemExtendedParameterAddressRequirements,
        PAGE_READWRITE, VirtualAlloc2,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut _,
        HighestEndingAddress: ((1usize << 37) - 1) as *mut _,
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: a fresh allocation; the parameter points at `requirements`,
    // which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null(),
            len,
            MEM_RESERVE | MEM_WRITE_WATCH | if commit { MEM_COMMIT } else { 0 },
            PAGE_READWRITE,
            &mut parameter,
            1,
        )
    } as usize;
    (at != 0).then_some(at)
}

impl Upload {
    pub(crate) fn new(len: usize) -> Option<Upload> {
        if !low() {
            return Some(Upload(Kind::Heap(vec![0; len])));
        }
        let n = len.max(1).div_ceil(BLOCK);
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        let found = arenas.iter().enumerate().find_map(|(a, arena)| Some((a, gap(&arena.taken, n)?)));
        let (a, first) = match found {
            Some(place) => place,
            None => {
                let blocks = n.max(ARENA / BLOCK);
                let at = reserve(blocks * BLOCK)?;
                crate::shadow::register(at, at as u64, (blocks * BLOCK) as u64);
                tracing::info!(at = format_args!("{at:#x}"), bytes = blocks * BLOCK, "memory for lists' uploads, where shaders reach it by address");
                arenas.push(Arena { at, taken: vec![false; blocks] });
                (arenas.len() - 1, 0)
            }
        };
        let arena = &mut arenas[a];
        arena.taken[first..first + n].fill(true);
        let at = arena.at + first * BLOCK;
        // SAFETY: the blocks are the arena's and nobody else's now.
        unsafe { std::ptr::write_bytes(at as *mut u8, 0, len) };
        Some(Upload(Kind::Low(at, len)))
    }

    pub(crate) fn start(&mut self) -> *mut u8 {
        match &mut self.0 {
            Kind::Low(at, _) => *at as *mut u8,
            Kind::Heap(block) => block.as_mut_ptr(),
        }
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        let Kind::Low(at, len) = self.0 else { return };
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(arena) = arenas.iter_mut().find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&at)) {
            let first = (at - arena.at) / BLOCK;
            arena.taken[first..first + len.max(1).div_ceil(BLOCK)].fill(false);
        }
    }
}

/// The block handed out that holds `address`: its start and bytes (whole
/// [`BLOCK`]s).
pub(crate) fn block(address: usize) -> Option<(usize, usize)> {
    let arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
    let arena = arenas.iter().find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&address))?;
    let first = (address - arena.at) / BLOCK;
    let n = arena.taken[first..].iter().take_while(|t| **t).count();
    (n > 0).then_some((arena.at + first * BLOCK, n * BLOCK))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_lie_below_40_bits_and_are_used_again_zeroed() {
        let mut a = Upload::new(100).unwrap();
        let mut b = Upload::new(BLOCK + 1).unwrap();
        let (pa, pb) = (a.start(), b.start());
        assert!((pa as usize) >= 1 << 32 && (pb as usize) + 2 * BLOCK <= 1 << 40);
        assert!((pa as usize).is_multiple_of(BLOCK) && pa != pb);
        // SAFETY: 100 bytes were handed out.
        unsafe { pa.write_bytes(0xab, 100) };
        drop(a);
        // The lowest free place (another test may have taken it meanwhile;
        // zeroed either way).
        let mut c = Upload::new(100).unwrap();
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(c.start(), 100) }.iter().all(|&x| x == 0));
        // More than an arena.
        let mut d = Upload::new(ARENA + 1).unwrap();
        // SAFETY: the bytes were handed out.
        unsafe { d.start().add(ARENA).write(1) };
        drop(b);
    }

    #[test]
    fn reservations_are_cut_from_the_window_once_it_is_there() {
        reserve_window();
        let (start, _) = *WINDOW_AT.lock().unwrap();
        assert!(start >= 1 << 32 && start + WINDOW <= 1 << 37);
        let a = reserve(100).unwrap();
        let b = reserve(BLOCK + 1).unwrap();
        assert!((start..start + WINDOW).contains(&a) && (start..start + WINDOW).contains(&b));
        assert!(a.is_multiple_of(BLOCK) && b.is_multiple_of(BLOCK) && a != b);
        // SAFETY: committed and zeroed, ours alone.
        assert!(unsafe { std::slice::from_raw_parts(b as *const u8, BLOCK + 1) }.iter().all(|&x| x == 0));
        // SAFETY: as above.
        unsafe { (b as *mut u8).add(BLOCK).write(1) };
    }

    #[test]
    fn a_gap_is_blocks_side_by_side() {
        assert_eq!(gap(&[true, false, true, false, false, true], 1), Some(1));
        assert_eq!(gap(&[true, false, true, false, false, true], 2), Some(3));
        assert_eq!(gap(&[true, false, true, false, false, true], 3), None);
    }
}
