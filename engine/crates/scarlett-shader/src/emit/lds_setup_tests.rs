//! A thread group's LDS set up by its first thread (`SCARLETT_LDS_FIRST_INIT`,
//! [`super::lds_setup`]) in the paired form (`bitcode::paired`): written by the group's first
//! thread whether its lane runs or not. Ported from Steenax6/VIctory-NVID ea2ba27c and
//! 2f16bc85 (GPL-3.0-or-later), whose tests ran on that tree's own paired form; here on
//! [`super::emit_paired`]. The helpers are that tree's `wave64_tests.rs` ones.
#![allow(dead_code)]

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ir, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// [`native_of`] with `lds` bytes of LDS (512-byte granules).
fn native_with(body: &[u32], wave64: bool, threads: u32, lds: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | lds.div_ceil(512) << 15, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

// Operands as the hardware numbers them: SGPR n is n, VCC 106 (its high dword 107), EXEC 126,
// an inline integer n 128 + n (-1: 193), a literal (the next dword) 255, VGPR n 256 + n.
const VCC: u32 = 106;
const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

const fn ds(op: u32, vdst: u32, addr: u32, data: u32, offset: u32) -> [u32; 2] {
    [0xD800_0000 | op << 18 | offset, vdst << 24 | data << 8 | addr]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// `shader` on the hardware device: one group, root constants `s0..s3`, a raw UAV of `dwords`
/// zeroed dwords, which come back; None (and a line) without a device that takes it.
fn run_paired(shader: &HostShader, constants: [u32; 4], dwords: u32) -> Option<Vec<u32>> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program: {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let mut out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(dwords)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out.swap_remove(0))
}

/// Panics at the first dword of `got` that is not the console's (`want`), naming the thread
/// and which of its four dwords.
fn same_as_console(got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len());
    if let Some(n) = (0..want.len()).find(|&n| got[n] != want[n]) {
        panic!("thread {}, dword {}: {:#x}, the console's {:#x}", n / 4, n % 4, got[n], want[n]);
    }
}

// ---- a thread group's LDS set up by its first thread (`SCARLETT_LDS_FIRST_INIT`) ----

/// The group's first thread setting up LDS dwords 0 and 1 (to 0 and 7) as the water's
/// transmission volumetrics do: `v_cmpx_eq_u32 exec, 0, v0`, the address and the data moved in,
/// `ds_write2_b32`.
const SETUP: [u32; 5] = [
    vopc(0xD2, k(0), 0),   // v_cmpx_eq_u32 exec, 0, v0
    vop1(0x01, 1, k(0)),   // v_mov_b32 v1, 0
    vop1(0x01, 2, k(7)),   // v_mov_b32 v2, 7
    0xD800_0000 | 0x0E << 18 | 0x100, // ds_write2_b32 v1, v1, v2 offset1:1
    2 << 16 | 1 << 8 | 1,
];

/// A wave64 program of one console wave (`threads` in its group, 256 bytes of LDS) whose
/// threads below s0 (by x) leave first; then `setup` under EXEC saved in s[14:15] and restored
/// after; then every lane left takes its x into dword 0 with `ds_max_u32` and adds 1 to dword
/// 1 with `ds_add_u32`, reads both back and stores them, with its x.
fn setup_program(setup: &[u32], threads: [u32; 3]) -> NativeProgram {
    let mut w = vec![
        vopc(0xC3, 0, 0),     // v_cmp_le_u32 vcc, s0, v0: the threads from s0 on
        sop1(0x24, 12, VCC),  // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, 0),        // s_cbranch_execz to the end (set below)
        sop1(0x04, 14, EXEC), // s_mov_b64 s[14:15], exec
    ];
    w.extend_from_slice(setup);
    w.extend([
        sop1(0x04, EXEC, 14), // s_mov_b64 exec, s[14:15]
        vop1(0x01, 3, k(0)),  // v_mov_b32 v3, 0
        vop1(0x01, 6, k(1)),  // v_mov_b32 v6, 1
    ]);
    w.extend(ds(0x08, 0, 3, 0, 0)); // ds_max_u32 v3, v0
    w.extend(ds(0x00, 0, 3, 6, 4)); // ds_add_u32 v3, v6 offset:4
    w.extend([0xD800_0000 | 0x37 << 18 | 0x100, 4 << 24 | 3]); // ds_read2_b32 v[4:5], v3 offset1:1
    w.extend(stores([4, 5, 0, 0]));
    w[2] = sopp(0x08, (w.len() - 3) as i16);
    let mut native = native_with(&w, true, threads.iter().product(), 256);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.threads = threads;
        // (thread ids in v0 and v1 for a group of rows)
        cs.rsrc2 |= u32::from(threads[1] > 1) << 11;
    }
    native
}

/// `native` in the paired form with the first thread's set-up on or off, validated by
/// `dxil.dll` when present; with the set-ups written by the first thread alone.
fn paired_setups(native: &NativeProgram, on: bool) -> (HostShader, u32) {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let bytes = super::container::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let (host, first) = (lower::TEST_HOST_WAVE.replace(Some(32)), super::lds_setup::TEST_ON.replace(Some(on)));
    super::lds_setup::TEST_SETUPS.set(0);
    let shader = super::emit_paired(&c, native, &p, &st);
    let setups = super::lds_setup::TEST_SETUPS.get();
    lower::TEST_HOST_WAVE.set(host);
    super::lds_setup::TEST_ON.set(first);
    let (shader, wave) = shader.expect("the paired form");
    assert_eq!(wave, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    (shader, setups)
}

/// The first thread's set-up is written once (in the lower lane's pass), by the first thread
/// whether its lane has left or not: on the card the lanes left read the dwords as set up and
/// as they took theirs into them, also where thread 0 and others left first. With
/// `SCARLETT_LDS_FIRST_INIT=0` the write stays the lane's, as before.
#[test]
fn the_groups_first_thread_sets_its_lds_up_also_where_its_lane_has_left() {
    let native = setup_program(&SETUP, [64, 1, 1]);
    let (on, setups) = paired_setups(&native, true);
    assert_eq!(setups, 1, "one set-up, by the first thread");
    let (off, none) = paired_setups(&native, false);
    assert_eq!(none, 0, "switched off: the lane's write");
    assert!(on.dxil != off.dxil);
    // (what `super::emit` gives the host)
    let shader = paired(&native);
    assert!(shader.dxil == on.dxil, "emit's choice: the paired form with the set-up");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for s0 in [0u32, 1, 5, 40] {
        let want: Vec<u32> = (0..64).flat_map(|t| if t < s0 { [0; 4] } else { [63, 7 + 64 - s0, t, t] }).collect();
        let Some(got) = run_paired(&shader, [s0, 0, 0, 0], 256) else { return };
        same_as_console(&got, &want);
    }
    eprintln!("ran on the hardware device: LDS set up by the first thread, its lane there or not");
}

/// A group of 8 rows of 8 whose test is x | y (`v_or_b32`) is the set-up; near misses are left
/// as the lane's write: a compare with 1, an address from an SGPR, data moved in before the
/// compare, and a group of rows tested by x alone.
#[test]
fn only_the_first_threads_set_up_is_written_by_the_first_thread_alone() {
    let mut rows = vec![vop2(0x1C, 7, v(1), 0)]; // v_or_b32 v7, v1, v0
    rows.push(vopc(0xD2, k(0), 7)); // v_cmpx_eq_u32 exec, 0, v7
    rows.extend_from_slice(&SETUP[1..]);
    assert_eq!(paired_setups(&setup_program(&rows, [8, 8, 1]), true).1, 1, "x | y in a group of rows");
    let mut one = SETUP;
    one[0] = vopc(0xD2, k(1), 0); // v_cmpx_eq_u32 exec, 1, v0
    let mut scalar = SETUP;
    scalar[1] = vop1(0x01, 1, 1); // v_mov_b32 v1, s1
    let early = [SETUP[2], SETUP[0], SETUP[1], SETUP[3], SETUP[4]];
    for (setup, threads, what) in [
        (&one[..], [64, 1, 1], "a compare with 1"),
        (&scalar[..], [64, 1, 1], "an address from an SGPR"),
        (&early[..], [64, 1, 1], "data moved in before the compare"),
        (&SETUP[..], [8, 8, 1], "a group of rows tested by x alone"),
    ] {
        assert_eq!(paired_setups(&setup_program(setup, threads), true).1, 0, "{what}");
    }
}

/// The set-up in each arm of a per-lane if/else whose arms both use LDS (`if (x < s0) { set
/// up; max } else { set up; max }`): the lanes missing at either set-up are in the other arm,
/// not gone, and the console's second arm starts from what the first left when the first
/// thread went the other way. Left as the lane's write.
#[test]
fn a_set_up_in_one_arm_of_a_per_lane_branch_is_the_lanes_write() {
    let mut arm = vec![sop1(0x04, 14, EXEC)]; // s_mov_b64 s[14:15], exec
    arm.extend_from_slice(&SETUP);
    arm.push(sop1(0x04, EXEC, 14)); // s_mov_b64 exec, s[14:15]
    arm.extend(ds(0x08, 0, 3, 0, 0)); // ds_max_u32 v3, v0
    let mut w = vec![
        vop1(0x01, 3, k(0)),  // v_mov_b32 v3, 0
        vopc(0xC4, 0, 0),     // v_cmp_gt_u32 vcc, s0, v0: x below s0
        sop1(0x24, 12, VCC),  // s_and_saveexec_b64 s[12:13], vcc
        sopp(0x08, arm.len() as i16), // s_cbranch_execz to the other arm
    ];
    w.extend_from_slice(&arm);
    w.extend([
        sop2(0x15, EXEC, 12, EXEC),         // s_andn2_b64 exec, s[12:13], exec: the other arm
        sopp(0x08, arm.len() as i16),       // s_cbranch_execz past it
    ]);
    w.extend_from_slice(&arm);
    w.push(sop1(0x04, EXEC, 12)); // s_mov_b64 exec, s[12:13]
    w.extend(stores([3, 3, 3, 3]));
    assert_eq!(paired_setups(&native_with(&w, true, 64, 256), true).1, 0);
}

/// `native` in the paired form (what `super::emit` gives a program with no exact form of its
/// own), validated by `dxil.dll` when present.
fn paired(native: &NativeProgram) -> HostShader {
    paired_setups(native, super::lds_setup::on()).0
}
