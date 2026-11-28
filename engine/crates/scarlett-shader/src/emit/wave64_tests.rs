//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    lowered_counted(native, p, wide, host_wave, false)
}

fn lowered_counted(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
    counted: bool,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower_counted(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    }, counted);
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// "Some lane is left" of a mask operation as a number: an upscaler's programs wait for
/// other groups in a loop that ends when every lane saw its flag, or after enough rounds:
/// both tests become -1 or 0 and SCC of `timeout | ~waiting` ends it.
#[test]
fn some_lane_of_a_mask_is_a_number_the_whole_wave_agrees_on() {
    let native = native(&[
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x7D84_0200, // v_cmp_eq_u32 vcc, s0, v1
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
        0x8592_80C1, // s_cselect_b64 s[18:19], -1, 0
        0xBF09_8501, // s_cmp_ge_u32 s1, 5
        0x8594_80C1, // s_cselect_b64 s[20:21], -1, 0
        0x8B92_1214, // s_orn2_b64 s[18:19], s[20:21], s[18:19]
        0x8506_8081, // s_cselect_b32 s6, 1, 0
        0x7E04_0206, // v_mov_b32 v2, s6
        0xE070_0000,
        0x8002_0200, // buffer_store_dword v2, off, s[8:11], 0
    ]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = on_64_lanes(&native, &p);
    // (lane s0 waits, if there is one; s1 rounds)
    for (wait, rounds, done) in [(0, 0, 0), (37, 4, 0), (63, 4, 0), (64, 0, 1), (37, 5, 1), (64, 9, 1)] {
        let Some(out) = dispatched(&shader, &[wait, rounds]) else {
            return;
        };
        assert_eq!(out, done, "lane {wait} after {rounds} rounds");
    }
}

/// `native` for a host of 64-lane waves.
fn on_64_lanes(native: &NativeProgram, p: &ir::Program) -> HostShader {
    let st = structure::structure(p).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(64));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered.expect("lower");
    assert_eq!(lowered.wave_size, Some(64));
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Dword 0 of the UAV after `shader` ran with root constants `constants`; none without a
/// device that takes it.
fn dispatched(shader: &HostShader, constants: &[u32]) -> Option<u32> {
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let (root, pso) = match host.compute_pipeline(shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 64?): {e}; not run");
            return None;
        }
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if u.root == RootType::Constants {
            ud[u.index as usize] = constants.get(u.dword as usize).copied().unwrap_or(0);
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// The same sum inside `if (t >= s0) { ... }` on a host of 64-lane waves: the lanes below s0
/// are elsewhere, and count as +0 in the console's tree.
#[test]
fn a_sum_of_halves_where_not_every_lane_runs_on_a_wave_of_64() {
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
        0x7D86_0000, // v_cmp_le_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | 23, // s_cbranch_execz to the restore of EXEC
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = on_64_lanes(&native, &p);
    let wide = |h: u16| match h {
        0 => 0.0,
        h => f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13),
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    for from in [0u32, 1, 5, 31, 32, 37, 63] {
        let mut tree: Vec<u16> = (0..64u16).map(|t| if u32::from(t) < from { 0 } else { 0x3C00 + 37 * t }).collect();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| narrow(wide(c[0]) + wide(c[1]))).collect();
        }
        let Some(out) = dispatched(&shader, &[from]) else {
            return;
        };
        assert_eq!(out, u32::from(tree[0]), "lanes from {from}");
    }
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `SCARLETT_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(ud))],
                views: &[View::RawUav(1)],
                samplers: &[],
                groups: [1, 1, 1],
            },
        )
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// A dispatch counted in threads: of the last group only the lanes below the
/// count store (after DexNom/ScarlettPort bc3cd4b0).
#[test]
fn the_lanes_past_a_count_of_threads_do_not_run() {
    // Each lane that runs adds 1 to counter[local x].
    let native = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7e04_0281, // v_mov_b32 v2, 1
        0xe0c8_1000, 0x8002_0201, // buffer_atomic_add v2, v1, s[8:11], 0 offen
    ]);
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let shader = lowered_counted(&native, &p, &HashSet::new(), 64, true).unwrap();
    let limit = shader
        .user_data
        .iter()
        .find(|u| u.root == RootType::Other(super::USER_DISPATCH_THREADS_X))
        .unwrap();
    assert_eq!((limit.slot, limit.dword), (0, 64));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else { return };
    let Ok((root, pso)) = host.compute_pipeline(&shader) else { return };
    for count in [0u32, 1, 7, 63, 64, 65, 74, 77, 127, 128] {
        let mut constants = vec![0; shader.user_data_dwords() as usize];
        constants[limit.index as usize] = count;
        let result = host
            .dispatch(&root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                views: &[View::RawUav(68)],
                samplers: &[],
                groups: [2, 1, 1],
            })
            .unwrap();
        let active = if count == 0 { 128 } else { count };
        for (i, &word) in result[0].iter().enumerate().take(64) {
            let i = i as u32;
            assert_eq!(word, u32::from(i < active) + u32::from(i + 64 < active), "count {count}, word {i}");
        }
    }
}

/// A console wave32 program with console lanes (`v_mbcnt`) in a 32 x 2 group: thread (x, y)
/// stores 1 at dword 32y + x. Every dword must be written once: the host's relabelled thread
/// ids are a bijection of the group. A 32-lane host need not give a host wave a 32-aligned run
/// of flat ids (2-D groups are filled from 16 x 2 tiles on NVIDIA and RDNA), so the first flat
/// id of a wave rounded down gave two waves the same ids (issue #1243; port of trustno1-coder/VIctory-NVIDIA 2dc3df1c); the waves now take
/// their threads from a group counter. With `SCARLETT_THREAD_TICKET=0` (the old rounding) the
/// count of dwords written is printed, not checked: it depends on the card.
#[test]
fn the_console_threads_of_a_two_dimensional_group_are_each_given_once() {
    let words = [
        0xD765_0003, // v_mbcnt_lo_u32_b32 v3, exec_lo, 0
        0x0001_007E,
        0x3408_0285, // v_lshlrev_b32 v4, 5, v1
        0x4A08_0104, // v_add_nc_u32 v4, v4, v0
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen
    ];
    let mut native = native_of(&words, false, 64);
    native.io = StageIo::Cs(ComputeIo {
        rsrc2: 5 << 1 | 1 << 11, // 5 user SGPRs, thread ids x and y in v0, v1
        threads: [32, 2, 1],
    });
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let lower_with = |ticket: bool| {
        // SAFETY: tests that read this switch hold ONE_AT_A_TIME
        unsafe { std::env::set_var("SCARLETT_THREAD_TICKET", if ticket { "1" } else { "0" }) };
        let shader = lowered_on(&native, &p, &HashSet::new(), 32);
        unsafe { std::env::remove_var("SCARLETT_THREAD_TICKET") };
        shader.expect("lower")
    };
    let shader = lower_with(true);
    let old = lower_with(false);
    assert_ne!(shader.dxil, old.dxil, "the counter is used for a group of two host waves");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let run = |shader: &HostShader| -> Option<Vec<u32>> {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the hardware device does not take this program: {e}; not run");
                return None;
            }
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        Some(out[0][..64].to_vec())
    };
    let Some(got) = run(&shader) else { return };
    let missing: Vec<usize> = (0..64).filter(|&i| got[i] != 1).collect();
    assert!(missing.is_empty(), "dwords not written once: {missing:?}");
    if let Some(before) = run(&old) {
        let written = before.iter().filter(|&&d| d == 1).count();
        eprintln!("the old rounding on this device: {written} of 64 dwords written");
    }
    eprintln!("ran on the hardware device: 64 of 64 console threads, each once");
}

const MASK_SCC_CASES: [(u32, u32); 9] = [
    (0, 0),
    (1, 0),
    (32, 31),
    (32, 32),
    (48, 7),
    (48, 37),
    (48, 49),
    (64, 63),
    (64, 64),
];

/// With two host waves for one console wave, one active lane is enough to set SCC
/// for all lanes still in EXEC. For example, with 48 active lanes and lane 37
/// selected, all 48 store 2; with lane 49 selected they all store 1.
#[derive(Clone, Copy, Debug)]
enum SccUse {
    Combined,
    Branch,
    Select,
}

fn nested_mask_scc(which: SccUse) -> NativeProgram {
    let choice: &[u32] = match which {
        SccUse::Combined => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0206, // v_mov_b32 v2, s6
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Branch => &[
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0282, // v_mov_b32 v2, 2
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Select => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0x7E04_0206, // v_mov_b32 v2, s6
        ],
    };
    let mut body = vec![
        0xBF8C_0000, // s_waitcnt 0: wait for the fixture's scalar descriptor load
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | (5 + choice.len() as u32), // s_cbranch_execz to the restore
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
    ];
    body.extend_from_slice(choice);
    body.extend_from_slice(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0201, // buffer_store_dword v2, v1, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&body)
}

fn emit_nested_mask_scc(native: &NativeProgram, p: &ir::Program) -> HostShader {
    let st = structure::structure(p).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(p, native).expect("io");
    let c = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let normal = super::emit(&c, native, p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    normal.expect("the production Wave32 fallback must lower the program")
}

#[test]
fn the_mask_scc_in_a_lane_branch_lowers_without_its_exec_skip() {
    let native = nested_mask_scc(SccUse::Combined);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
    lower::take_needs();
    let direct = lowered(&native, &p, &HashSet::new());
    let needs = lower::take_needs();
    assert!(matches!(
        direct,
        Err(ShaderError::Unsupported { what, .. }) if lower::needs_every_lane(what)
    ));
    let st = structure::structure(&p).expect("structure");
    let enclosing = lower::branches_around(&st, &needs).expect("enclosing lane branch");
    assert_eq!(enclosing.len(), 1);
    assert_eq!(
        p.blocks[enclosing[0]].insts.last().unwrap().name(),
        "s_cbranch_execz"
    );
    let without_skip = ir::without_skips(&p, false).expect("removable EXEC skip");
    lowered(&native, &without_skip, &HashSet::new()).expect("without skips");
    let shader = emit_nested_mask_scc(&native, &p);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation");
    }
}

/// Use SCARLETT_WARP=1 to run the same readback on WARP.
#[test]
#[ignore = "requires a D3D12 device that executes Wave32"]
fn both_halves_observe_the_same_mask_scc_on_a_wave32_device() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).expect("Wave32 adapter");
    // SAFETY: the LUID is the one returned by this device, and DXGI fills its adapter description.
    let desc = unsafe {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        let factory: IDXGIFactory4 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("DXGI");
        factory
            .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
            .unwrap()
            .GetDesc1()
            .unwrap()
    };
    let end = desc
        .Description
        .iter()
        .position(|&w| w == 0)
        .unwrap_or(desc.Description.len());
    eprintln!(
        "Wave32 device: {}",
        String::from_utf16_lossy(&desc.Description[..end])
    );

    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the feature structure is initialized and sized as required by D3D12.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
            .expect("wave capabilities");
    }
    eprintln!(
        "Wave range: {}..{}",
        caps.WaveLaneCountMin, caps.WaveLaneCountMax
    );
    if !caps.WaveOps.as_bool() || !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&32) {
        eprintln!("0 Wave32 lane comparisons; unsupported on this adapter");
        return;
    }
    let mut total_wrong = 0;
    for which in [SccUse::Combined, SccUse::Branch, SccUse::Select] {
        eprintln!("SCC_VARIANT {which:?}");
        let native = nested_mask_scc(which);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
        let shader = emit_nested_mask_scc(&native, &p);
        let (root, pso) = host.compute_pipeline(&shader).expect("Wave32 pipeline");
        for (below, selected) in MASK_SCC_CASES {
            let mut constants = vec![0; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants && u.slot == 0 && u.dword < 2 {
                    constants[u.index as usize] = [below, selected][u.dword as usize];
                }
            }
            let buffers = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(constants))],
                        views: &[View::RawUav(64)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch and readback");
            let result = &buffers[0];
            assert_eq!(result.len(), 64);
            eprintln!("SCC_WORDS {which:?} {below} {selected} {result:?}");
            let mut wrong = 0;
            for (lane, &word) in result.iter().enumerate() {
                let expected = if lane as u32 >= below {
                    0
                } else {
                    1 + u32::from(selected < below)
                };
                if word != expected {
                    if wrong == 0 {
                        eprintln!(
                            "below {below}, selected {selected}: lane {lane} read {word}, expected {expected}"
                        );
                    }
                    wrong += 1;
                }
            }
            eprintln!("below {below}, selected {selected}: {wrong}/64 differences");
            total_wrong += wrong;
        }
    }
    assert_eq!(total_wrong, 0, "mismatching words in the generated cases");
}

/// An independent DXC-compiled group-wide OR. Compile the accompanying HLSL with
/// `dxc -T cs_6_6 -E main -Fo ref.dxil testdata/wave64-mask-scc.hlsl` and set
/// `SCC_REFERENCE_DXIL` to that file before explicitly running this test.
#[test]
#[ignore = "requires a D3D12 device and the compiled SCC_REFERENCE_DXIL"]
fn independent_hlsl_group_any_scc_oracle() {
    use windows::Win32::Graphics::Direct3D12::ID3D12RootSignature;
    let host = Host::new(false).expect("adapter");
    let code = std::fs::read(std::env::var_os("SCC_REFERENCE_DXIL").unwrap()).unwrap();
    // SAFETY: our compiled shader embeds its root signature.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &code) }.unwrap();
    let pso = host.compute_pipeline_with(&root, &code).unwrap();
    for (below, selected) in MASK_SCC_CASES {
        let buffers = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(vec![below, selected])),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 0,
                            },
                        ),
                    ],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let data = &buffers[0];
        let expected: Vec<u32> = (0..64)
            .map(|i| {
                if i >= below {
                    0
                } else {
                    1 + u32::from(selected < below)
                }
            })
            .collect();
        eprintln!("SCC_WORDS Hlsl {below} {selected} {data:?}");
        assert_eq!(data, &expected);
    }
}

const MASK_SCC_CASES: [(u32, u32); 9] = [
    (0, 0),
    (1, 0),
    (32, 31),
    (32, 32),
    (48, 7),
    (48, 37),
    (48, 49),
    (64, 63),
    (64, 64),
];

/// With two host waves for one console wave, one active lane is enough to set SCC
/// for all lanes still in EXEC. For example, with 48 active lanes and lane 37
/// selected, all 48 store 2; with lane 49 selected they all store 1.
#[derive(Clone, Copy, Debug)]
enum SccUse {
    Combined,
    Branch,
    Select,
}

fn nested_mask_scc(which: SccUse) -> NativeProgram {
    let choice: &[u32] = match which {
        SccUse::Combined => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0206, // v_mov_b32 v2, s6
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Branch => &[
            0xBF84_0002, // s_cbranch_scc0 to the value 1
            0x7E04_0282, // v_mov_b32 v2, 2
            0xBF82_0001, // s_branch to the store
            0x7E04_0281, // v_mov_b32 v2, 1
        ],
        SccUse::Select => &[
            0x8506_8182, // s_cselect_b32 s6, 2, 1
            0x7E04_0206, // v_mov_b32 v2, s6
        ],
    };
    let mut body = vec![
        0xBF8C_0000, // s_waitcnt 0: wait for the fixture's scalar descriptor load
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0000 | (5 + choice.len() as u32), // s_cbranch_execz to the restore
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0x8792_7E6A, // s_and_b64 s[18:19], vcc, exec
    ];
    body.extend_from_slice(choice);
    body.extend_from_slice(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xE070_1000,
        0x8002_0201, // buffer_store_dword v2, v1, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&body)
}

fn emit_nested_mask_scc(native: &NativeProgram, p: &ir::Program) -> HostShader {
    let st = structure::structure(p).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(p, native).expect("io");
    let c = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let normal = super::emit(&c, native, p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    normal.expect("the production Wave32 fallback must lower the program")
}

#[test]
fn the_mask_scc_in_a_lane_branch_lowers_without_its_exec_skip() {
    let native = nested_mask_scc(SccUse::Combined);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
    lower::take_needs();
    let direct = lowered(&native, &p, &HashSet::new());
    let needs = lower::take_needs();
    assert!(matches!(
        direct,
        Err(ShaderError::Unsupported {
            what: lower::DIVERGENT_MASK_EXCHANGE,
            ..
        })
    ));
    let st = structure::structure(&p).expect("structure");
    let enclosing = lower::branches_around(&st, &needs).expect("enclosing lane branch");
    assert_eq!(enclosing.len(), 1);
    assert_eq!(
        p.blocks[enclosing[0]].insts.last().unwrap().name(),
        "s_cbranch_execz"
    );
    let without_skip = ir::without_skips(&p, false).expect("removable EXEC skip");
    lowered(&native, &without_skip, &HashSet::new()).expect("without skips");
    let shader = emit_nested_mask_scc(&native, &p);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation");
    }
}

/// Use SCARLETT_WARP=1 to run the same readback on WARP.
#[test]
#[ignore = "requires a D3D12 device that executes Wave32"]
fn both_halves_observe_the_same_mask_scc_on_a_wave32_device() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).expect("Wave32 adapter");
    // SAFETY: the LUID is the one returned by this device, and DXGI fills its adapter description.
    let desc = unsafe {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        let factory: IDXGIFactory4 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("DXGI");
        factory
            .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
            .unwrap()
            .GetDesc1()
            .unwrap()
    };
    let end = desc
        .Description
        .iter()
        .position(|&w| w == 0)
        .unwrap_or(desc.Description.len());
    eprintln!(
        "Wave32 device: {}",
        String::from_utf16_lossy(&desc.Description[..end])
    );

    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the feature structure is initialized and sized as required by D3D12.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
            .expect("wave capabilities");
    }
    eprintln!(
        "Wave range: {}..{}",
        caps.WaveLaneCountMin, caps.WaveLaneCountMax
    );
    if !caps.WaveOps.as_bool() || !(caps.WaveLaneCountMin..=caps.WaveLaneCountMax).contains(&32) {
        eprintln!("0 Wave32 lane comparisons; unsupported on this adapter");
        return;
    }
    let mut total_wrong = 0;
    for which in [SccUse::Combined, SccUse::Branch, SccUse::Select] {
        eprintln!("SCC_VARIANT {which:?}");
        let native = nested_mask_scc(which);
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("IR");
        let shader = emit_nested_mask_scc(&native, &p);
        let (root, pso) = host.compute_pipeline(&shader).expect("Wave32 pipeline");
        for (below, selected) in MASK_SCC_CASES {
            let mut constants = vec![0; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants && u.slot == 0 && u.dword < 2 {
                    constants[u.index as usize] = [below, selected][u.dword as usize];
                }
            }
            let buffers = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(constants))],
                        views: &[View::RawUav(64)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch and readback");
            let result = &buffers[0];
            assert_eq!(result.len(), 64);
            eprintln!("SCC_WORDS {which:?} {below} {selected} {result:?}");
            let mut wrong = 0;
            for (lane, &word) in result.iter().enumerate() {
                let expected = if lane as u32 >= below {
                    0
                } else {
                    1 + u32::from(selected < below)
                };
                if word != expected {
                    if wrong == 0 {
                        eprintln!(
                            "below {below}, selected {selected}: lane {lane} read {word}, expected {expected}"
                        );
                    }
                    wrong += 1;
                }
            }
            eprintln!("below {below}, selected {selected}: {wrong}/64 differences");
            total_wrong += wrong;
        }
    }
    assert_eq!(total_wrong, 0, "mismatching words in the generated cases");
}

/// An independent DXC-compiled group-wide OR. Compile the accompanying HLSL with
/// `dxc -T cs_6_6 -E main -Fo ref.dxil testdata/wave64-mask-scc.hlsl` and set
/// `SCC_REFERENCE_DXIL` to that file before explicitly running this test.
#[test]
#[ignore = "requires a D3D12 device and the compiled SCC_REFERENCE_DXIL"]
fn independent_hlsl_group_any_scc_oracle() {
    use windows::Win32::Graphics::Direct3D12::ID3D12RootSignature;
    let host = Host::new(false).expect("adapter");
    let code = std::fs::read(std::env::var_os("SCC_REFERENCE_DXIL").unwrap()).unwrap();
    // SAFETY: our compiled shader embeds its root signature.
    let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &code) }.unwrap();
    let pso = host.compute_pipeline_with(&root, &code).unwrap();
    for (below, selected) in MASK_SCC_CASES {
        let buffers = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(vec![below, selected])),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 0,
                            },
                        ),
                    ],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let data = &buffers[0];
        let expected: Vec<u32> = (0..64)
            .map(|i| {
                if i >= below {
                    0
                } else {
                    1 + u32::from(selected < below)
                }
            })
            .collect();
        eprintln!("SCC_WORDS Hlsl {below} {selected} {data:?}");
        assert_eq!(data, &expected);
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
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = super::Container::parse(&bytes).expect("container");
    let (host, first) = (lower::TEST_HOST_WAVE.replace(Some(32)), super::lds_setup::TEST_ON.replace(Some(on)));
    super::lds_setup::TEST_SETUPS.set(0);
    let shader = super::paired(&c, native, &p, &io, &res);
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
