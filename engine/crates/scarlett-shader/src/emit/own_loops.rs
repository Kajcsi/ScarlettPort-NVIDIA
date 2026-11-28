//! Loops each lane of a console wave may leave on its own.
//!
//! The console's compiler closes a loop that each lane runs for its own count with a branch
//! on "some lane is left": the lanes still to go are a mask, each pass leaves out of it the
//! lanes that are done, and the wave goes round again while the mask has a bit:
//!
//! ```text
//!   left = EXEC
//!   loop:  EXEC = before & left          ; the lanes that are done sit the pass out
//!          before = EXEC
//!          ...                           ; the pass
//!          EXEC &= this lane is done
//!          left &= ~EXEC                 ; SCC: some lane is left
//!          branch to loop on SCC
//! ```
//!
//! The branch is the whole wave's (`Lower::is_wave_scc_branch`). A console wave64 program
//! in two host waves of 32 takes it by exchanging the halves' answers through LDS, a
//! meeting of the two waves, twice every pass (`Lower::wave_wide`); and since every lane
//! then runs the pass, an LDS access in it gets meetings of its own. The title's ray tile
//! sort spends most of a frame (140 ms of 250 in scar16) in such meetings: its count and
//! scatter loops do one LDS atomic per pass.
//!
//! It does not have to. A lane that is done does nothing in any later pass: the pass starts
//! by taking EXEC within the mask of the lanes left, and EXEC is only narrowed after that.
//! So the lane may as well leave the loop when its own bit of the mask goes, as with a branch
//! on its own bit: then each host wave runs as many passes as its own lanes need, and the
//! halves meet once before the loop and once after it.
//!
//! [`find`] proves that for each such branch, with no instruction sequence taken on trust:
//!
//! - the branch closes a loop: one way goes back into the blocks reachable from it that do
//!   not post-dominate it (its region, [`super::lower::lane_regions`]), from which the branch
//!   is reached again; the other way, the only way out, post-dominates it; the loop has one
//!   entry from outside and no exit of the program;
//! - the block the branch goes back to is entered from the branch alone and starts, before
//!   any vector instruction or read of EXEC, by setting EXEC to something ANDed with the very
//!   mask whose SCC the branch reads;
//! - every other write of EXEC in the loop narrows it (`v_cmpx`, an AND or AND-NOT of EXEC,
//!   `s_and_saveexec`) or sets it back to a copy of EXEC made in the loop that no pass's
//!   end lies between (not in the blocks between the branch and the loop's entry made
//!   before them);
//! - the loop has no cross-lane step (`v_mbcnt`, `v_readlane`, `v_writelane`,
//!   `v_readfirstlane`, permutes, swizzles, DPP), no LDS access but reads and folds (the
//!   atomics whose order does not change their sum, `ds_add_u32`...), no scalar store or
//!   atomic, barrier, message or end, and reads no lane mask as data;
//! - no scalar or mask value it writes (EXEC included) is read after it: those count the
//!   wave's passes, not the lane's.
//!
//! The loop's blocks are then not run by every lane ([`super::lower`]'s `lds_meetings` puts
//! the meetings at their edges, and refuses an LDS read and write in them as before). Any
//! other loop keeps the wave-wide branch. `SCARLETT_WAVE_OWN_LOOPS=0` turns this off;
//! `SCARLETT_WAVE_WHY=1` says, for each such branch, what refused it.

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::structure::{LaneExit, Structured};

type Why = &'static str;

const NOT_A_LOOP: Why = "not the latch of a loop left only while no lane is left";
const NO_NARROWING: Why = "a pass that does not start by leaving out the lanes that are done";
const WIDENS: Why = "a write of EXEC that may give a lane that is done its bit back";
const CROSS: Why = "a cross-lane step";
const LDS: Why = "an LDS access other than a read or a fold";
const EFFECT: Why = "a scalar effect, a barrier, a message or an end";
const MASK: Why = "a lane mask read as data";
const LEAVES: Why = "a scalar or mask value of the loop read after it";

/// What [`find`] proved.
#[derive(Debug, Default, Clone)]
pub(super) struct OwnLoops {
    /// The blocks closed by such a branch: each lane takes it by its own bit.
    pub latches: HashSet<usize>,
    /// Every block of those loops, the latches included: not every lane runs them.
    pub blocks: HashSet<usize>,
}

#[cfg(test)]
thread_local! {
    /// Tests: [`on`] as given, whatever the environment says.
    pub(super) static TEST_ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Whether [`find`] is used (`SCARLETT_WAVE_OWN_LOOPS=0`: not).
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_ON.get() {
        return on;
    }
    std::env::var("SCARLETT_WAVE_OWN_LOOPS").map_or(true, |v| v != "0")
}

/// The branches on "some lane is left" of `p` each lane may take by its own bit, with their
/// loops' blocks. `why`: one line per such branch on standard error, with what refused it.
pub(super) fn find(p: &Program, st: &Structured, why: bool) -> OwnLoops {
    let mut out = OwnLoops::default();
    let Some(regions) = super::lower::lane_regions(st) else {
        return out;
    };
    let mut def: HashMap<Value, (usize, usize)> = HashMap::new();
    for (b, blk) in p.blocks.iter().enumerate() {
        for (i, inst) in blk.insts.iter().enumerate() {
            for &(_, w) in &inst.writes {
                def.insert(w, (b, i));
            }
        }
    }
    for (d, region) in regions {
        let Some(left) = mask_scc_branch(p, st, d, &def) else {
            continue;
        };
        match check(p, st, d, left, &region) {
            Ok(blocks) => {
                if why {
                    eprintln!("WAVE64-OWN-LOOP b{d}: each lane leaves on its own");
                }
                out.latches.insert(d);
                out.blocks.extend(blocks);
            }
            Err(w) => {
                if why {
                    eprintln!("WAVE64-OWN-LOOP b{d}: {w}");
                }
            }
        }
    }
    out
}

/// The mask (its low dword) whose SCC closes block `d`, if `d` ends in an `s_cbranch_scc0`
/// or `scc1` on a mask operation's SCC that does not write EXEC (as
/// `Lower::is_wave_scc_branch` takes them) and is not taken wave-wide anyway.
fn mask_scc_branch(p: &Program, st: &Structured, d: usize, def: &HashMap<Value, (usize, usize)>) -> Option<Value> {
    let last = p.blocks.get(d)?.insts.last()?;
    if !matches!(last.name(), "s_cbranch_scc0" | "s_cbranch_scc1") || st.wide.contains(&d) {
        return None;
    }
    let scc = last.read(Reg::Scc)?;
    if !st.scc_some.contains_key(&scc) {
        return None;
    }
    let &(b, i) = def.get(&scc)?;
    let by = &p.blocks[b].insts[i];
    if by.write(Reg::S(EXEC)).is_some() {
        return None;
    }
    let Fields::Sop { dst: Some(Operand::Scalar { code, .. }), .. } = &by.fields else {
        return None;
    };
    by.write(Reg::S(*code))
}

/// The scalar register an SOP operand names, with the value it holds there.
fn scalar_src(inst: &Inst, k: usize) -> Option<(u8, Option<Value>)> {
    let Fields::Sop { src, .. } = &inst.fields else {
        return None;
    };
    match src.get(k)? {
        &Operand::Scalar { code, .. } => Some((code, inst.read(Reg::S(code)))),
        _ => None,
    }
}

fn is_vector(inst: &Inst) -> bool {
    !matches!(inst.fields, Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. })
}

fn check(p: &Program, st: &Structured, d: usize, left: Value, region: &[bool]) -> Result<Vec<usize>, Why> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End | LaneExit::Indirect => {}
        }
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, s) in succ.iter().enumerate() {
        for &t in s {
            preds[t].push(b);
        }
    }
    let LaneExit::Branch { taken, not_taken, .. } = st.exits[d] else {
        return Err(NOT_A_LOOP);
    };
    let any = st.wave_any.get(d).copied().flatten().ok_or(NOT_A_LOOP)?;
    // the way the wave goes while some lane is left, and the way out
    let (back, out) = if any { (taken, not_taken) } else { (not_taken, taken) };
    if back == out || !region.get(back).copied().unwrap_or(false) || region.get(out).copied().unwrap_or(true) {
        return Err(NOT_A_LOOP);
    }
    // the loop: the region and the branch, which the region goes back to
    if !(0..n).any(|b| region[b] && succ[b].contains(&d)) {
        return Err(NOT_A_LOOP);
    }
    let mut r = region.to_vec();
    r[d] = true;
    for b in (0..n).filter(|&b| r[b]) {
        if matches!(st.exits[b], LaneExit::End | LaneExit::Indirect) {
            return Err(EFFECT);
        }
        if succ[b].iter().any(|&t| !r[t] && t != out) {
            return Err(NOT_A_LOOP);
        }
    }
    let entries: Vec<usize> = (0..n).filter(|&b| r[b] && preds[b].iter().any(|&q| !r[q])).collect();
    let [entry] = entries[..] else {
        return Err(NOT_A_LOOP);
    };
    if back == entry || preds[back] != [d] {
        return Err(NO_NARROWING);
    }
    // the blocks between the branch and the loop's entry
    let mut between = vec![false; n];
    let mut stack = vec![back];
    while let Some(b) = stack.pop() {
        if b == entry || !r[b] || between[b] {
            continue;
        }
        between[b] = true;
        stack.extend(succ[b].iter().copied());
    }
    // the pass starts by taking EXEC within the lanes left
    let mut narrowed = false;
    for inst in &p.blocks[back].insts {
        if inst.write(Reg::S(EXEC)).is_some() {
            let within = inst.name() == "s_and_b64"
                && (0..2).any(|k| scalar_src(inst, k).is_some_and(|(_, v)| v == Some(left)));
            if !within {
                return Err(NO_NARROWING);
            }
            narrowed = true;
            break;
        }
        if is_vector(inst) || inst.read(Reg::S(EXEC)).is_some() {
            return Err(NO_NARROWING);
        }
    }
    if !narrowed {
        return Err(NO_NARROWING);
    }
    // copies of EXEC made in the loop, by the block that makes them
    let mut copies: HashMap<Value, usize> = HashMap::new();
    for b in (0..n).filter(|&b| r[b]) {
        for inst in &p.blocks[b].insts {
            let name = inst.name();
            let Fields::Sop { dst: Some(Operand::Scalar { code, .. }), .. } = &inst.fields else {
                continue;
            };
            let copy = name == "s_mov_b64" && scalar_src(inst, 0).is_some_and(|(c, _)| c == EXEC)
                || name.ends_with("_saveexec_b64");
            if copy && *code != EXEC && let Some(v) = inst.write(Reg::S(*code)) {
                copies.insert(v, b);
            }
        }
    }
    let restores = |b: usize, v: Option<Value>| {
        v.and_then(|v| copies.get(&v).copied()).is_some_and(|made| !between[b] || between[made])
    };
    for b in (0..n).filter(|&b| r[b]) {
        for inst in &p.blocks[b].insts {
            let name = inst.name();
            match &inst.fields {
                Fields::Sopp { .. } => {
                    if name == "s_barrier"
                        || name.starts_with("s_sendmsg")
                        || name.starts_with("s_endpgm")
                        || name == "s_trap"
                        || name == "s_sethalt"
                    {
                        return Err(EFFECT);
                    }
                    continue;
                }
                Fields::Smem { .. } if name.contains("store") || name.contains("atomic") => return Err(EFFECT),
                Fields::Ds { gds, .. } => {
                    if *gds || !(name.starts_with("ds_read") || super::lower::fold_identity(name).is_some()) {
                        return Err(LDS);
                    }
                }
                Fields::Buffer { lds: true, .. } | Fields::Flat { lds: true, .. } => return Err(LDS),
                Fields::Valu(v) if v.dpp.is_some() => return Err(CROSS),
                _ => {}
            }
            if name.starts_with("v_mbcnt")
                || name.starts_with("v_readlane")
                || name.starts_with("v_writelane")
                || name.starts_with("v_readfirstlane")
                || name.starts_with("v_permlane")
            {
                return Err(CROSS);
            }
            if super::lower::reads_mask_as_data(st, inst) {
                return Err(MASK);
            }
            if inst.write(Reg::S(EXEC)).is_none() && inst.write(Reg::S(EXEC + 1)).is_none() {
                continue;
            }
            let src = |k| scalar_src(inst, k);
            let is_exec = |k| src(k).is_some_and(|(c, _)| c == EXEC);
            let ok = name.starts_with("v_cmpx")
                || matches!(name, "s_and_saveexec_b64" | "s_andn1_saveexec_b64")
                || name == "s_and_b64" && (is_exec(0) || is_exec(1))
                || name == "s_andn2_b64" && is_exec(0)
                || name == "s_and_b64" && (0..2).any(|k| src(k).is_some_and(|(_, v)| v == Some(left)))
                || name == "s_mov_b64" && restores(b, src(0).and_then(|s| s.1))
                || name == "s_or_b64"
                    && (is_exec(0) && restores(b, src(1).and_then(|s| s.1))
                        || is_exec(1) && restores(b, src(0).and_then(|s| s.1)));
            if !ok {
                return Err(WIDENS);
            }
        }
    }
    // no scalar or mask value of the loop is read after it (carried on by joins or not)
    let mut mine: HashSet<Value> = HashSet::new();
    for b in (0..n).filter(|&b| r[b]) {
        let blk = &p.blocks[b];
        mine.extend(blk.phis.iter().filter(|f| matches!(f.reg, Reg::S(_) | Reg::Scc)).map(|f| f.value));
        for inst in &blk.insts {
            mine.extend(inst.writes.iter().filter(|w| matches!(w.0, Reg::S(_) | Reg::Scc)).map(|w| w.1));
        }
    }
    loop {
        let before = mine.len();
        for b in (0..p.blocks.len()).filter(|&b| !r.get(b).copied().unwrap_or(false)) {
            for f in &p.blocks[b].phis {
                if f.args.iter().any(|a| mine.contains(a)) {
                    mine.insert(f.value);
                }
            }
        }
        if mine.len() == before {
            break;
        }
    }
    for b in (0..p.blocks.len()).filter(|&b| !r.get(b).copied().unwrap_or(false)) {
        if p.blocks[b].insts.iter().any(|i| i.reads.iter().any(|(_, v)| mine.contains(v))) {
            return Err(LEAVES);
        }
    }
    Ok((0..n).filter(|&b| r[b]).collect())
}
