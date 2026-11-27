//! A stand-in for the indirect light the title traces rays for.
//!
//! With the ray-tracing structure builder off (`crate::pipeline::builder_runs`)
//! the title's rays find nothing. Its program that resolves the diffuse light
//! of its screen probes (`CS_ScreenProbes_TemporalResolve_*`) then leaves the
//! sky's light outdoors, nothing indoors (walls are black where no lamp or
//! sun reaches) and, in blocks of the picture's tiles, what the probes'
//! memory held. Right after that program the light is raised to an estimate
//! and bounded by a multiple of it.
//!
//! That a stand-in can be put there, and in which of the program's images,
//! is beloooo's find (ScarlettPort-Belobug-AMD, 06031237, 4f45ab88, cd9f78ce,
//! GPL-3.0-or-later), which puts a constant there, shaded in creases from the
//! scene's depth. The estimate here is worked out from the frame instead, so
//! it follows the hour, the room and its lamps by itself: the light the
//! visible surfaces send out (the title keeps it for its own screen traces:
//! `CS_CacheScreenSpaceRadiance`, the frame before) is averaged in 64x36
//! cells, and every point gets the mean of the cells weighted by the share of
//! its surroundings they cover, as a directional value (first-order harmonics
//! at 160x90 points, taken between them by depth).
//!
//! What it assumes: what is not on screen sends out what is (with a slow
//! mean of the last seconds, so that turning into a dark corner is not black
//! at once). No occlusion but the creases'.
//!
//! `SCARLETT_AMBIENT`:
//! - unset or `0`/`off`: the title's own output.
//! - a number: that share of the estimate (1: all of it).
//! - `constant:<level>`: the constant, as beloooo's fork has it.
//!
//! `SCARLETT_AMBIENT_SHADE=<strength>,<radius in metres>` sets the creases.
//!
//! The air of a room is lit by the same probes: without rays they hold the
//! sky's light indoors too, a brown veil at sunset. `SCARLETT_AMBIENT_AIR`
//! bounds what the air scatters by a multiple of the light around.
//!
//! Translucent surfaces (the material that lets light through: paper,
//! posters, lamp shades) get a light of their own from the program that
//! gathers the probes' light at each pixel (`CS_FinalGather_*`): what the
//! lamps and the sun send through them from behind, kept in an image of its
//! own (R32_UINT holding R9G9B9E5, or floats) that the tiled lighting adds.
//! With the title's ray-traced light (`SCARLETT_BVH_BUILDER=1`) such surfaces
//! look right; without it that light reaches them through walls too, a
//! violet glow indoors. So without rays that image is cleared right after
//! the program, whether the stand-in is set or not. The cost: translucent
//! surfaces get no light from behind (lamp shades, curtains against the
//! sun); what falls on their front stays. `SCARLETT_TRANSLUCENT_BACKLIGHT=1`
//! keeps the title's.
//!
//! Reflections are traced too, at half the picture's size: rays the
//! title's screen-space march resolves (`CS_ResolveScreenSpace`) take the
//! light seen on screen; the others go to the structures, find nothing, and
//! `CS_RayMiss` gives each the sky's light from the sky's cube maps (in a
//! cutscene frame, capture haze2: 10 to 30 against 0.01 to 0.1 for the room
//! the eyes and floors are in, and blue). `SCARLETT_AMBIENT_SPECULAR` bounds,
//! right after that program, what it wrote by the same estimate taken along
//! the reflected direction ([`Role::Miss`]); the screen-space resolve that
//! comes next still writes what it found over it.

use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host};

/// Programs of the title the stand-in follows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Role {
    /// Resolves the diffuse light.
    Resolve,
    /// Keeps the light visible surfaces send out.
    Radiance,
    /// Adds the lamps' light to what the air scatters (the last program
    /// that writes that volume before it is summed up along the view).
    Air,
    /// Gives the reflection rays that found nothing the sky's light
    /// (`CS_RayMiss`, not `CS_RayMiss_Volumetrics` nor the screen probes').
    Miss,
    /// Gathers the probes' light at each pixel, and the light translucent
    /// surfaces get from behind.
    Gather,
}

/// The program's part in this, by its name in the container.
pub(crate) fn role(container: &[u8]) -> Option<Role> {
    let has = |name: &[u8]| container.windows(name.len()).any(|w| w == name);
    if has(b"CS_ScreenProbes_TemporalResolve") {
        Some(Role::Resolve)
    } else if has(b"CS_CacheScreenSpaceRadiance") {
        Some(Role::Radiance)
    } else if has(b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(Role::Air)
    } else if is_miss(container) {
        Some(Role::Miss)
    } else if has(b"CS_FinalGather") {
        Some(Role::Gather)
    } else {
        None
    }
}

/// `CS_RayMiss_` and the eight hex digits of a variant's hash.
fn is_miss(container: &[u8]) -> bool {
    const NAME: &[u8] = b"CS_RayMiss_";
    container.windows(NAME.len() + 9).any(|w| {
        w.starts_with(NAME) && w[NAME.len()..NAME.len() + 8].iter().all(u8::is_ascii_hexdigit) && w[NAME.len() + 8] == b'_'
    })
}

/// Where the resolve writes the diffuse light (R9G9B9E5 as 32-bit integers)
/// and reads the scene's depth, and where the other writes the light of
/// visible surfaces: root parameter and place in its table, in dwords.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const RADIANCE: (u8, u32) = (1, 16);
/// Where the light the air scatters is.
pub(crate) const AIR: (u8, u32) = (1, 32);
/// Where the miss program writes the reflected light (R11G11B10, half the
/// picture's size) and where it reads the depth it is given (a 32-bit depth
/// of the picture's size).
pub(crate) const MISS_LIGHT: (u8, u32) = (1, 104);
pub(crate) const MISS_DEPTH: (u8, u32) = (0, 1000);
/// The miss program's first root constant: which surfaces its rays are of.
pub(crate) const MISS_OPAQUE: u32 = 0;
pub(crate) const MISS_ALPHA: u32 = 1;
/// Where the gather writes the light translucent surfaces get from behind:
/// root parameter, place in its table, and the program's register for it.
/// Packed (R9G9B9E5 in an R32_UINT image) at the first when the title's
/// setting asks for it, as in the frames seen; as floats at the second when
/// it does not. Both name the same image in those frames, and the program
/// writes nothing but zeros or that light at either.
pub(crate) const BACKLIGHT: [(u8, u32, u32); 2] = [(1, 120, 32), (1, 112, 36)];

/// How the image at a place of [`BACKLIGHT`] (its dword in the table) is
/// cleared, by its format: as integers, as floats, or not at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Form {
    Packed,
    Float,
}

pub(crate) fn backlight_form(dword: u32, format: i32) -> Option<Form> {
    match (dword, format) {
        // R32_UINT
        (120 | 112, 42) => Some(Form::Packed),
        // R32G32B32A32_FLOAT, R16G16B16A16_FLOAT, R11G11B10_FLOAT, R32_FLOAT
        (112, 2 | 10 | 26 | 41) => Some(Form::Float),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Set {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn parse(set: &str) -> Set {
    let set = set.trim();
    let number = |t: &str| t.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0);
    match set.strip_prefix("constant:") {
        Some(level) => number(level).map_or(Set::Off, Set::Constant),
        None => number(set).map_or(Set::Off, Set::Estimate),
    }
}

fn set() -> Set {
    static SET: OnceLock<Set> = OnceLock::new();
    *SET.get_or_init(|| parse(&std::env::var("SCARLETT_AMBIENT").unwrap_or_default()))
}

/// Whether there is something to do after a program of that role: the
/// title's rays find nothing, and the stand-in is set (after the gather:
/// the title's light from behind is not asked for).
pub(crate) fn on(role: Role) -> bool {
    if role == Role::Gather {
        return !backlight_kept() && !crate::pipeline::builder_runs();
    }
    (set() != Set::Off || specular().is_some()) && !crate::pipeline::builder_runs() && !off_now()
}

/// `SCARLETT_AMBIENT_SPECULAR=<share>[,<sky>]`: reflections whose rays found
/// nothing are made no brighter than 1.5 times `share` of the estimate along
/// the reflected direction (the sky's light they get is 100 to 1000 times the
/// room's); `sky` (0 to 1, 0 if not given) of them stays the title's sky as
/// it was. Unset or 0: the title's sky.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Specular {
    share: f32,
    sky: f32,
}

fn parse_specular(set: &str) -> Option<Specular> {
    let mut parts = set.split(',').map(|t| t.trim().parse::<f32>().ok().filter(|v| v.is_finite()));
    let share = parts.next().flatten().filter(|v| *v > 0.0)?;
    let sky = parts.next().flatten().map_or(0.0, |v| v.clamp(0.0, 1.0));
    Some(Specular { share, sky })
}

fn specular() -> Option<Specular> {
    static SET: OnceLock<Option<Specular>> = OnceLock::new();
    *SET.get_or_init(|| parse_specular(&std::env::var("SCARLETT_AMBIENT_SPECULAR").unwrap_or_default()))
}

/// `SCARLETT_TRANSLUCENT_BACKLIGHT=1`: translucent surfaces keep the light
/// the title gives them from behind, though without rays it comes through walls
/// (live with `SCARLETT_AB=<s>,SCARLETT_TRANSLUCENT_BACKLIGHT`, diagnostics:
/// on, the title's).
fn backlight_kept() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("SCARLETT_TRANSLUCENT_BACKLIGHT") {
        return on;
    }
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| kept_by(&std::env::var("SCARLETT_TRANSLUCENT_BACKLIGHT").unwrap_or_default()))
}

fn kept_by(set: &str) -> bool {
    matches!(set.trim(), "1" | "on")
}

/// `SCARLETT_AB=<s>,SCARLETT_AMBIENT_OFF` (diagnostics): the stand-in and the
/// air's bound are off in the phases that have it, the title's own output
/// stands (as `SCARLETT_AMBIENT=0`, `SCARLETT_AMBIENT_AIR=0`).
fn off_now() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        crate::ab::pick("SCARLETT_AMBIENT_OFF").unwrap_or(false)
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

/// The most the air scatters, as a multiple of the light around
/// (`SCARLETT_AMBIENT_AIR=<multiple>`; unset or 0: what the title has).
fn air() -> f32 {
    static SET: OnceLock<f32> = OnceLock::new();
    *SET.get_or_init(|| {
        let set = std::env::var("SCARLETT_AMBIENT_AIR").unwrap_or_default();
        set.trim().parse().ok().filter(|v: &f32| v.is_finite() && *v > 0.0).unwrap_or(0.0)
    })
}

/// How the slow mean of the light around follows the frames
/// (`SCARLETT_AMBIENT_SETTLE=<time constant in ms>[,<jump factor>[,<fresh
/// seconds>]]`, or `SCARLETT_AMBIENT_SETTLE`; `1`: 250,3; unset or 0: 0.04 of
/// the way a frame, as before, which is 25 frames: 0.8 s at the console's 30
/// frames a second, 3-4 s at 7-9 here, the haze that hangs in a room after
/// walking in). Set, it is the same share of the way a second whatever the
/// frame rate, and a change by more than the jump factor (default 3; 0: none)
/// is taken at once. Fresh seconds (default 0: none): for that long after
/// such a jump the light the title's own resolve left is bounded by the
/// estimate, not four times it. Live with `SCARLETT_AB=<s>,SCARLETT_AMBIENT_SETTLE`
/// (on: the value asked, or 250,3). From avatarmanz/ScarlettPort-7900XTX
/// b4ba4bbb and 9576c45f, without the light probe.
#[derive(Clone, Copy)]
struct Settle {
    ms: f32,
    jump: f32,
    fresh: f32,
}

fn settle() -> Option<Settle> {
    static SET: OnceLock<Option<Settle>> = OnceLock::new();
    let asked = *SET.get_or_init(|| {
        let var = std::env::var("SCARLETT_AMBIENT_SETTLE").or_else(|_| std::env::var("SCARLETT_AMBIENT_SETTLE"));
        #[cfg(feature = "diagnostics")]
        let var = if var.is_err() && crate::ab::bit("SCARLETT_AMBIENT_SETTLE").is_some() { Ok("1".to_owned()) } else { var };
        let set = var.ok()?;
        let set = if set.trim() == "1" { "250,3".to_owned() } else { set };
        let mut parts = set.split(',').map(|p| p.trim().parse::<f32>().ok());
        let ms = parts.next().flatten().filter(|v| v.is_finite() && *v > 0.0)?;
        let jump = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(3.0);
        let fresh = parts.next().flatten().filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(0.0);
        tracing::info!(ms, jump, fresh, "the stand-in's mean of the light around settles by the clock (SCARLETT_AMBIENT_SETTLE)");
        Some(Settle { ms, jump, fresh })
    });
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("SCARLETT_AMBIENT_SETTLE") {
        return asked.filter(|_| on);
    }
    asked
}

/// The share of the way the slow mean goes at this frame's keeping
/// ([`Role::Radiance`]; the other passes do not read it).
fn follow(role: Role, settle: Option<Settle>) -> f32 {
    match settle {
        Some(settle) if role == Role::Radiance => 1.0 - (-frame_ms(role) / settle.ms).exp(),
        _ => 0.04,
    }
}

/// Milliseconds since the last keeping ([`Role::Radiance`], once a frame);
/// the first, and after a pause of over a second, as a frame of 30. Other
/// roles: the last answer.
fn frame_ms(role: Role) -> f32 {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
    static LAST: AtomicU64 = AtomicU64::new(0);
    static DT: AtomicU32 = AtomicU32::new(0x4204_0000); // 33.0
    if role != Role::Radiance {
        return f32::from_bits(DT.load(Relaxed));
    }
    // (Asked twice in a keeping: the words and the share.)
    static ASKED: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    if ASKED.swap(now, Relaxed) == now {
        return f32::from_bits(DT.load(Relaxed));
    }
    let last = LAST.swap(now, Relaxed);
    let dt = match now.saturating_sub(last) {
        0 => 1.0,
        dt if last == 0 || dt > 1000 => 33.0,
        dt => dt as f32,
    };
    DT.store(dt.to_bits(), Relaxed);
    dt
}

/// Strength and radius of the creases' shade.
fn shade() -> [f32; 2] {
    static SET: OnceLock<[f32; 2]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6];
        let set = std::env::var("SCARLETT_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=15, b0), DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), UAV(u1), UAV(u2)"
#define CX 64
#define CY 36
#define CELLS (CX * CY)
#define PX 160
#define PY 90
// Where the points' distances are in `points` (after three a point).
#define DEPTHS (3 * PX * PY)
cbuffer Set : register(b0) {
    float level; float constant; float strength; float radius;
    float near; float tan_y; uint width; uint height;
    float follow; float most; float air; float flat;
    float jump; float fresh; float dt;
};
Texture2D<float> depth : register(t0);
// Cells: the light they send out (then, at CELLS, the slow mean), from
// CELLS + 1 on where they are.
RWStructuredBuffer<float4> cells : register(u1);
// Three a point: red, green and blue, each a mean and a direction.
RWStructuredBuffer<float4> points : register(u2);
static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
// With `flat` (no depth among the program's views but a stencil plane) every
// point is taken 4 m away: the estimate is the light around, without creases.
float3 seen(int2 p, float2 size) {
    float z = 4.0;
    if (flat == 0.0) {
        float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
        z = min(near / max(d, 1e-7), 1000.0);
    }
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}

#ifdef KEEP
RWTexture2D<float3> radiance : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void keep(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    radiance.GetDimensions(w, h);
    float3 sum = 0;
    for (uint y = 0; y < 5; y++) for (uint x = 0; x < 5; x++) {
        uint2 p = uint2((float2(at.xy) + (float2(x, y) + 0.5) / 5.0) * float2(w, h) / float2(CX, CY));
        float3 c = max(radiance[p], 0.0);
        c = any(isnan(c)) || any(isinf(c)) ? 0.0 : c;
        // Lamps and glints count, but not for everything.
        sum += c / max(1.0, dot(c, LUMA) / 20.0);
    }
    cells[at.y * CX + at.x] = float4(sum / 25.0, 1.0);
}
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void mean() {
    float3 sum = 0;
    for (uint i = 0; i < CELLS; i++) sum += cells[i].rgb;
    float4 was = cells[CELLS];
    float3 now = sum / CELLS;
    // (`jump` > 0: a change of more than that factor either way is
    // followed at once, as a new room is.)
    float a = max(dot(now, LUMA), 1e-5), b = max(dot(was.rgb, LUMA), 1e-5);
    bool jumped = jump > 0.0 && max(a / b, b / a) > jump;
    // Its fourth part: seconds since the light around last jumped (from
    // 0.001; 0: nothing kept yet).
    float age = was.a > 0.0 && !jumped ? was.a + dt : 0.001;
    cells[CELLS] = float4(was.a > 0.0 ? lerp(was.rgb, now, jumped ? 1.0 : follow) : now, age);
}
#endif

#ifdef PLACE
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void place(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(CX, CY)), size);
    cells[CELLS + 1 + at.y * CX + at.x] = float4(p, p.z * p.z);
}
#endif

#ifdef GATHER
[RootSignature(RS)]
[numthreads(8, 2, 1)]
void gather(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    float3 p = seen(int2((float2(at.xy) + 0.5) * size / float2(PX, PY)), size);
    float r2 = 0.25 + 0.0025 * p.z * p.z;
    float whole = 0;
    float3 toward = 0, light = 0, red = 0, green = 0, blue = 0;
    for (uint i = 0; i < CELLS; i++) {
        float4 c = cells[CELLS + 1 + i];
        float3 L = cells[i].rgb;
        float3 d = c.xyz - p;
        float dd = dot(d, d);
        float w = c.w / (dd + r2);
        float3 dir = d / sqrt(dd + 0.25 * r2);
        whole += w;
        toward += w * dir;
        light += w * L;
        red += w * L.r * dir;
        green += w * L.g * dir;
        blue += w * L.b * dir;
    }
    whole = max(whole, 1e-6);
    light /= whole;
    toward /= whole;
    uint at0 = (at.y * PX + at.x) * 3;
    points[at0] = float4(red / whole - light.r * toward, light.r);
    points[at0 + 1] = float4(green / whole - light.g * toward, light.g);
    points[at0 + 2] = float4(blue / whole - light.b * toward, light.b);
    // Its distance, after the three of every point (for the air's bound).
    points[DEPTHS + at.y * PX + at.x] = float4(p.z, 0.0, 0.0, 0.0);
}
#endif

#ifdef AIR
RWTexture3D<float3> scattered : register(u0);
// No more than a multiple of the light around: of the room as the last
// seconds showed it, and of the surface seen in this direction (towards a
// window the air outside may be as bright as the sky is). That surface is
// the one behind the air: of the four points around the column, those as
// far as the farthest of them, between them as they lie. Taken from the
// nearest point whatever its distance, a lit figure in front of a dark wall
// let the lamps' light into the air behind it in blocks of the volume's
// columns around the figure (red steps around people in a bar, scar13).
[RootSignature(RS)]
[numthreads(8, 8, 2)]
void clear_air(uint3 at : SV_DispatchThreadID) {
    uint w, h, d;
    scattered.GetDimensions(w, h, d);
    if (at.x >= w || at.y >= h || at.z >= d) return;
    float2 g = (float2(at.xy) + 0.5) / float2(w, h) * float2(PX, PY) - 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float zk[4];
    float back = 0.0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        zk[k] = points[DEPTHS + q.y * PX + q.x].x;
        back = max(back, zk[k]);
    }
    float3 seen_there = 0.0;
    float sum = 0.0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float wgt = (b.x * b.y + 1e-3) * (zk[k] >= 0.7 * back ? 1.0 : 1e-3);
        uint at0 = (q.y * PX + q.x) * 3;
        seen_there += wgt * float3(points[at0].w, points[at0 + 1].w, points[at0 + 2].w);
        sum += wgt;
    }
    seen_there /= max(sum, 1e-6);
    float far = float(at.z) / float(d);
    float cap = air * lerp(dot(cells[CELLS].rgb, LUMA), dot(seen_there, LUMA), far);
    float3 c = scattered[at];
    float has = dot(c, LUMA);
    if (has > cap) scattered[at] = c * (cap / has);
}
#endif

#ifdef REFLECT
RWTexture2D<float3> reflected : register(u0);
// The light the points around send along the direction the view reflects
// to at the point `c` seen at `p` (as `raise` takes it along the normal),
// `level` its share.
float3 reflected_estimate(int2 p, float3 c, float2 size) {
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    float3 v = normalize(c);
    float3 ray = v - 2.0 * dot(v, n) * n;
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float wgt = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += wgt * points[at0];
        sg += wgt * points[at0 + 1];
        sb += wgt * points[at0 + 2];
        sum += wgt;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    // Radiance from the first moments: three times the direction's share
    // for an even spread of directions, a little less for a smooth image.
    float3 found = all + 2.0 * float3(dot(sr.xyz, ray), dot(sg.xyz, ray), dot(sb.xyz, ray)) / sum;
    found = clamp(found, 0.1 * all, 3.0 * all);
    return lerp(found, cells[CELLS].rgb, 0.2) * level;
}
// Each texel of the reflections' image (half the picture's size): what the
// title put there (the sky's light, for every ray that found nothing) is
// bounded by REFLECT_MOST times the estimate. Only what is brighter than
// that changes, towards the estimate's colour as the cut grows: no texel is
// made darker than the light around it or brighter than the title had it,
// and a texel the miss program did not write keeps its value. Replacing
// every texel by the estimate (pepezin's 2db59d48) went with rooms black and
// sparkling in run scar4. `constant` is the share of the title's value kept
// as it was.
#define REFLECT_MOST 1.5
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void reflect_light(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    reflected.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float3 was = reflected[at.xy];
    was = any(isnan(was)) || any(isinf(was)) ? 0.0 : max(was, 0.0);
    float has = dot(was, LUMA);
    float2 size = float2(width, height);
    int2 p = int2((float2(at.xy) + 0.5) * size / float2(w, h));
    float3 c = seen(p, size);
    // The sky at the texel's place in the picture (at half size, an edge
    // between a surface and the sky): the light around, as the last seconds
    // showed it. (Kept as the title had it, such edges sparkled with the
    // sky's light.)
    float3 found = c.z >= 999.0 ? cells[CELLS].rgb * level : reflected_estimate(p, c, size);
    found = any(isnan(found)) || any(isinf(found)) ? 0.0 : max(found, 0.0);
    float lf = dot(found, LUMA);
    float top = max(lf * REFLECT_MOST, 1e-6);
    float3 kept = was;
    if (has > top) {
        float3 dimmed = was * (top / has);
        float3 tinted = lf > 1e-6 ? found * (top / lf) : dimmed;
        kept = lerp(dimmed, tinted, saturate(log2(has / top) / 4.0));
    }
    reflected[at.xy] = lerp(kept, was, constant);
}
#endif

#ifdef RAISE
RWTexture2D<uint> light : register(u0);
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (top * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void raise(uint3 at : SV_DispatchThreadID) {
    float2 size = float2(width, height);
    if (at.x >= width || at.y >= height) return;
    int2 p = int2(at.xy);
    float3 c = seen(p, size);
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx) + float3(0, 0, -1e-9));
    if (dot(n, c) > 0.0) n = -n;
    // How open it is (as in beloooo's cd9f78ce).
    float open = 1.0;
    float px = min(radius * size.y / (2.0 * tan_y * c.z), 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    if (strength > 0.0 && px >= 1.5 && c.z < 400.0) {
        float sum = 0.0;
        const int N = 12;
        for (int i = 0; i < N; i++) {
            float a = i * 2.399963 + noise * 6.2831853;
            float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
            float3 v = seen(p + int2(round(o)), size) - c;
            float vv = dot(v, v);
            float over = dot(v, n) - 0.004 * c.z - 0.005;
            sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
        }
        open = saturate(1.0 - strength * 2.0 * sum / N);
    }
    float3 found;
    if (constant > 0.0) {
        found = float3(1.0, 0.98, 0.94) * constant * open;
        light[p] = pack(found);
        return;
    }
    // Between the four points around, those at its own depth first; which
    // four is dithered a little.
    float2 g = (float2(p) + 0.5) / size * float2(PX, PY) - 0.5 + (noise - 0.5) * 0.5;
    int2 g0 = int2(floor(g));
    float2 f = g - float2(g0);
    float4 sr = 0, sg = 0, sb = 0;
    float sum = 0;
    for (int k = 0; k < 4; k++) {
        int2 q = clamp(g0 + int2(k & 1, k >> 1), int2(0, 0), int2(PX, PY) - 1);
        float z = seen(int2((float2(q) + 0.5) * size / float2(PX, PY)), size).z;
        float2 b = float2(k & 1 ? f.x : 1 - f.x, k >> 1 ? f.y : 1 - f.y);
        float w = (b.x * b.y + 1e-3) / (0.02 + abs(z - c.z) / c.z);
        uint at0 = (q.y * PX + q.x) * 3;
        sr += w * points[at0];
        sg += w * points[at0 + 1];
        sb += w * points[at0 + 2];
        sum += w;
    }
    float3 all = float3(sr.w, sg.w, sb.w) / sum;
    found = all + 1.2 * float3(dot(sr.xyz, n), dot(sg.xyz, n), dot(sb.xyz, n)) / sum;
    found = clamp(found, 0.2 * all, 2.5 * all);
    found = lerp(found, cells[CELLS].rgb, 0.2) * level * open;
    float3 was = unpack(light[p]);
    // Magenta/violet garbage the probes' memory held (green far under red and
    // blue): dropped, as VIctory-Belobug-AMD does (ambient.rs, "Filter magenta
    // garbage probes"). Seen on an RTX 5080 as a purple-blue glow on a lit
    // sheet of paper in the prison cell; not an AMD-only fault.
    if (was.g < 0.5 * min(was.r, was.b) && max(was.r, was.b) > 0.02) was = 0.0;
    // And the bluer kind of it (blue over 2.5x green and over red): left as
    // a faint violet by the line above. The sky's light keeps green above
    // 0.4x blue, sunsets keep blue low, so neither is touched.
    if (was.b > 2.5 * was.g && was.b > was.r && was.b > 0.02) was = 0.0;
    // What the program left counts as far as it can be true: the sky's light
    // outdoors, not the blocks of other memory.
    // For `fresh` seconds after the light around jumped (a new room) the
    // title's leftover counts for no more than the estimate, then up to
    // `most` times it again: its history holds the place before.
    float bound = fresh > 0.0 ? lerp(1.0, most, saturate(cells[CELLS].a / fresh)) : most;
    float top = max(dot(found, LUMA) * bound, 1e-6);
    float has = dot(was, LUMA);
    was *= min(1.0, top / max(has, 1e-6));
    light[p] = pack(max(was, found));
}
#endif
"#;

/// Bytes of the cells' buffer and of the points'.
const CELLS: u64 = 64 * 36;
const CELL_BYTES: u64 = (2 * CELLS + 1) * 16;
/// Three float4 a point, then one more a point for its distance.
const POINT_BYTES: u64 = 160 * 90 * 64;

struct Pass {
    root: ID3D12RootSignature,
    keep: ID3D12PipelineState,
    mean: ID3D12PipelineState,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: ID3D12PipelineState,
    reflect: ID3D12PipelineState,
    cells: ID3D12Resource,
    points: ID3D12Resource,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device, buffer: impl Fn(u64) -> Option<ID3D12Resource>) -> windows::core::Result<Pass> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |part: &str, entry| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&format!("#define {part}\n{SOURCE}"), entry, s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let keep = pipeline("KEEP", s!("keep"))?;
    let mean = pipeline("KEEP", s!("mean"))?;
    let place = pipeline("PLACE", s!("place"))?;
    let gather = pipeline("GATHER", s!("gather"))?;
    let raise = pipeline("RAISE", s!("raise"))?;
    let air = pipeline("AIR", s!("clear_air"))?;
    let reflect = pipeline("REFLECT", s!("reflect_light"))?;
    let made = |bytes| buffer(bytes).ok_or_else(windows::core::Error::empty);
    Ok(Pass {
        root: root.ok_or_else(windows::core::Error::empty)?,
        keep,
        mean,
        place,
        gather,
        raise,
        air,
        reflect,
        cells: made(CELL_BYTES)?,
        points: made(POINT_BYTES)?,
    })
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        make(&host.device, |bytes| host.uav_buffer(bytes))
            .inspect_err(|e| tracing::warn!(error = %e, "no stand-in for the indirect light"))
            .ok()
    })
    .as_ref()
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER { Type: D3D12_RESOURCE_BARRIER_TYPE_UAV, ..Default::default() };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Whether the light of a frame has been kept: the estimate has something
/// to begin with.
static KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the passes read and write, as views in the heap the list has set.
pub(crate) struct Views {
    /// The image the title's program just wrote.
    pub(crate) image: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The scene's depth, for [`Role::Resolve`].
    pub(crate) depth: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The size of the image, and how many slices it has.
    pub(crate) size: [u32; 2],
    pub(crate) slices: u32,
    /// The size of the depth (the image's but for [`Role::Miss`]).
    pub(crate) depth_size: [u32; 2],
    /// `depth` is a stencil plane, no depth: every point is taken at one
    /// distance (the estimate is the light around, without creases).
    pub(crate) flat: bool,
}

/// For the line every ten seconds: what the stand-in for reflections did.
struct Said {
    /// Dispatches of the miss program it followed, frames they were in,
    /// and those it left as they were: of water, and before a frame's
    /// light was kept (or no tables).
    ran: AtomicU64,
    frames: AtomicU64,
    water: AtomicU64,
    not_ready: AtomicU64,
    last_frame: AtomicU64,
    /// Bit 0 opaque surfaces, 1 alpha, 2 water, 3 others.
    layers: AtomicU64,
}

static SAID_MISS: Said = Said {
    ran: AtomicU64::new(0),
    frames: AtomicU64::new(0),
    water: AtomicU64::new(0),
    not_ready: AtomicU64::new(0),
    last_frame: AtomicU64::new(u64::MAX),
    layers: AtomicU64::new(0),
};

/// What came of a dispatch of the miss program, counted for the line.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum MissNote {
    Ran(u32),
    Water,
    NotReady,
}

/// Counts what came of a dispatch of the miss program; every ten seconds an
/// info line says it, so a run shows that the stand-in ran and where.
pub(crate) fn note_miss(note: MissNote) {
    use AtomicOrdering::Relaxed;
    if specular().is_none() {
        return;
    }
    let s = &SAID_MISS;
    match note {
        MissNote::Ran(layer) => {
            s.ran.fetch_add(1, Relaxed);
            s.layers.fetch_or(1 << layer.min(3), Relaxed);
            let frame = crate::frame::current();
            if s.last_frame.swap(frame, Relaxed) != frame {
                s.frames.fetch_add(1, Relaxed);
            }
        }
        MissNote::Water => {
            s.water.fetch_add(1, Relaxed);
        }
        MissNote::NotReady => {
            s.not_ready.fetch_add(1, Relaxed);
        }
    }
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    let Some(since) = *last else {
        *last = Some(now);
        return;
    };
    if now.duration_since(since).as_secs_f32() < 10.0 {
        return;
    }
    *last = Some(now);
    let layers = s.layers.swap(0, Relaxed);
    let named: Vec<&str> = [(1, "opaque"), (2, "alpha"), (4, "water"), (8, "other")]
        .iter()
        .filter(|(bit, _)| layers & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    let Specular { share, sky } = specular().unwrap_or(Specular { share: 0.0, sky: 0.0 });
    tracing::info!(
        "a stand-in for reflections (SCARLETT_AMBIENT_SPECULAR={share},{sky}) in the last {:.0} s: \
         {} frames, {} dispatches of CS_RayMiss followed (surfaces: {}), each by an estimate along \
         the reflected direction put in right after it (before CS_ResolveScreenSpace); {} of water \
         and {} before a frame's light was kept left as the title had them",
        now.duration_since(since).as_secs_f32(),
        s.frames.swap(0, Relaxed),
        s.ran.swap(0, Relaxed),
        if named.is_empty() { "none".to_owned() } else { named.join(", ") },
        s.water.swap(0, Relaxed),
        s.not_ready.swap(0, Relaxed),
    );
}

/// After the title's program of that role, in `list` with the program's view
/// heap set. Replaces the list's compute root signature and pipeline; false
/// if it recorded nothing.
pub(crate) fn after(list: &HostList, role: Role, views: &Views) -> bool {
    use std::sync::atomic::Ordering;
    // (The gather has its own pass, [`clear_backlight`].)
    if role == Role::Gather {
        return false;
    }
    let Some(pass) = pass() else {
        return false;
    };
    // The share of the estimate, and the constant (for reflections: the
    // share of the title's sky kept); the keep runs for either stand-in.
    let (level, constant) = match (role, set(), specular()) {
        (Role::Miss, _, Some(s)) => (s.share, s.sky),
        (Role::Miss, _, None) => return false,
        (Role::Radiance, Set::Off, Some(_)) => (0.0, 0.0),
        (_, Set::Off, _) => return false,
        (_, Set::Estimate(level), _) => (level, 0.0f32),
        (_, Set::Constant(level), _) => (0.0, level),
    };
    let estimate = role == Role::Miss || constant == 0.0;
    if role != Role::Radiance && estimate && !KEPT.load(Ordering::Relaxed) {
        if role == Role::Miss {
            note_miss(MissNote::NotReady);
        }
        return false;
    }
    // The air's bound works with the constant too: the light around (the
    // cells) is kept in both forms, and the points seen are gathered for it
    // below. Without the bound the air held the sky's light indoors: orange
    // clouds that burst in the prison's cells and a brown veil in its halls.
    if role == Role::Air && air() == 0.0 {
        return false;
    }
    static SAID: Mutex<u32> = Mutex::new(0);
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if *said < 5 {
            *said += 1;
            tracing::info!(?role, size = ?views.size, level, constant, "a stand-in for the indirect light");
        }
    }
    list.keep(&pass.cells);
    list.keep(&pass.points);
    let [strength, radius] = shade();
    let floats = [level, constant, strength, radius, 0.1, 0.45];
    let mut words = [0u32; 15];
    for (to, v) in words.iter_mut().zip(floats) {
        *to = v.to_bits();
    }
    words[6] = views.depth_size[0];
    words[7] = views.depth_size[1];
    let settle = settle();
    words[8] = follow(role, settle).to_bits();
    words[9] = 4.0f32.to_bits();
    words[10] = air().to_bits();
    words[11] = if views.flat { 1.0f32 } else { 0.0f32 }.to_bits();
    words[12] = settle.map_or(0.0, |s| s.jump).to_bits();
    words[13] = settle.map_or(0.0, |s| s.fresh).to_bits();
    words[14] = (frame_ms(role) / 1000.0).to_bits();
    let l = &list.list;
    // SAFETY: an open list with the heap of the views set; the title's
    // program has just used both views, so their textures are in the states
    // these views need; buffers the list keeps.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&pass.root);
        l.SetComputeRoot32BitConstants(0, 15, words.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, views.depth);
        l.SetComputeRootDescriptorTable(2, views.image);
        l.SetComputeRootUnorderedAccessView(3, pass.cells.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, pass.points.GetGPUVirtualAddress());
        let run = |pso: &ID3D12PipelineState, x, y, z| {
            l.SetPipelineState(pso);
            l.Dispatch(x, y, z);
            uav_barrier(l);
        };
        match role {
            Role::Radiance => {
                run(&pass.keep, 8, 9, 1);
                run(&pass.mean, 1, 1, 1);
                KEPT.store(true, Ordering::Relaxed);
            }
            Role::Gather => {}
            Role::Air => run(&pass.air, views.size[0].div_ceil(8), views.size[1].div_ceil(8), views.slices.div_ceil(2)),
            Role::Resolve => {
                // (With the constant, only for the air's bound.)
                if constant == 0.0 || air() > 0.0 {
                    run(&pass.place, 8, 9, 1);
                    run(&pass.gather, 20, 45, 1);
                }
                run(&pass.raise, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
            Role::Miss => {
                // The points afresh, from this frame's depth: the diffuse
                // stand-in makes them later in the frame.
                run(&pass.place, 8, 9, 1);
                run(&pass.gather, 20, 45, 1);
                run(&pass.reflect, views.size[0].div_ceil(8), views.size[1].div_ceil(8), 1);
            }
        }
    }
    true
}

/// Clears the light translucent surfaces get from behind: its own small
/// pass, so that the stand-in's are not made for it when the stand-in is off.
const CLEAR_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), DescriptorTable(UAV(u0))"
cbuffer Set : register(b0) { uint width; uint height; };
#ifdef PACKED
RWTexture2D<uint> image : register(u0);
#else
RWTexture2D<float4> image : register(u0);
#endif
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void clear(uint3 at : SV_DispatchThreadID) {
    if (at.x < width && at.y < height) image[at.xy] = 0;
}
"#;

struct Clear {
    root: ID3D12RootSignature,
    packed: ID3D12PipelineState,
    float: ID3D12PipelineState,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clear {}
// SAFETY: as above.
unsafe impl Sync for Clear {}

fn make_clear(device: &ID3D12Device) -> windows::core::Result<Clear> {
    let mut root: Option<ID3D12RootSignature> = None;
    let mut pipeline = |defines: &str| -> windows::core::Result<ID3D12PipelineState> {
        let code = crate::host::compile(&format!("{defines}\n{CLEAR_SOURCE}"), s!("clear"), s!("cs_5_1"))?;
        // SAFETY: the root signature is inside the shader; plain creations.
        unsafe {
            if root.is_none() {
                root = Some(device.CreateRootSignature(0, &code)?);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr().cast(), BytecodeLength: code.len() },
                ..Default::default()
            };
            device.CreateComputePipelineState(&desc)
        }
    };
    let packed = pipeline("#define PACKED")?;
    let float = pipeline("")?;
    Ok(Clear { root: root.ok_or_else(windows::core::Error::empty)?, packed, float })
}

/// After the gather, in `list` with the program's view heap set: `image` (a
/// view in that heap, of `size`) to zeros. Replaces the list's compute root
/// signature and pipeline; false if it recorded nothing.
pub(crate) fn clear_backlight(
    list: &HostList,
    image: D3D12_GPU_DESCRIPTOR_HANDLE,
    size: [u32; 2],
    form: Form,
    dword: u32,
) -> bool {
    static CLEAR: OnceLock<Option<Clear>> = OnceLock::new();
    let made = || {
        let made = make_clear(&host()?.device);
        made.inspect_err(|e| tracing::warn!(error = %e, "translucent surfaces' light from behind is not cleared")).ok()
    };
    let Some(clear) = CLEAR.get_or_init(made).as_ref() else {
        return false;
    };
    static SAID: Mutex<Vec<(u32, Form)>> = Mutex::new(Vec::new());
    {
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if !said.contains(&(dword, form)) && said.len() < 4 {
            said.push((dword, form));
            tracing::info!(
                ?form,
                dword,
                ?size,
                "translucent surfaces' light from behind cleared after the final gather: without rays it comes through walls (SCARLETT_TRANSLUCENT_BACKLIGHT=1 keeps it)"
            );
        }
    }
    let l = &list.list;
    // SAFETY: an open list with the heap of the view set; the title's program
    // has just written through that view, so the image is in the state it needs.
    unsafe {
        uav_barrier(l);
        l.SetComputeRootSignature(&clear.root);
        l.SetComputeRoot32BitConstants(0, 2, size.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(1, image);
        l.SetPipelineState(if form == Form::Packed { &clear.packed } else { &clear.float });
        l.Dispatch(size[0].div_ceil(8), size[1].div_ceil(8), 1);
        uav_barrier(l);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings() {
        assert_eq!(parse(""), Set::Off);
        assert_eq!(parse("off"), Set::Off);
        assert_eq!(parse("0"), Set::Off);
        assert_eq!(parse("0.8"), Set::Estimate(0.8));
        assert_eq!(parse("constant:0.3"), Set::Constant(0.3));
        assert_eq!(parse("constant:nan"), Set::Off);
    }

    #[test]
    fn specular_settings() {
        assert_eq!(parse_specular(""), None);
        assert_eq!(parse_specular("0"), None);
        assert_eq!(parse_specular("off"), None);
        assert_eq!(parse_specular("1"), Some(Specular { share: 1.0, sky: 0.0 }));
        assert_eq!(parse_specular("0.6, 0.25"), Some(Specular { share: 0.6, sky: 0.25 }));
        assert_eq!(parse_specular("0.6,7"), Some(Specular { share: 0.6, sky: 1.0 }));
        assert_eq!(parse_specular("0.6,x"), Some(Specular { share: 0.6, sky: 0.0 }));
    }

    #[test]
    fn roles_by_name() {
        assert_eq!(role(b"..CS_ScreenProbes_TemporalResolve_a911b49c_Wrapped.."), Some(Role::Resolve));
        assert_eq!(role(b"..CS_CacheScreenSpaceRadiance_eed2fc35.."), Some(Role::Radiance));
        assert_eq!(role(b"CS_AccumulateLocalLightInscatteringVolume_7e2f5eef"), Some(Role::Air));
        assert_eq!(role(b"CS_TemporalResolve"), None);
        assert_eq!(role(b"..CS_RayMiss_de6b91a6_Wrapped.."), Some(Role::Miss));
        assert_eq!(role(b"..CS_RayMiss_Volumetrics_3c5d8c00_Wrapped.."), None);
        assert_eq!(role(b"..CS_ScreenProbe_RayMiss_80a88544_Wrapped.."), None);
        // The two programs seen with the light from behind are listed as
        // CS_FinalGather_1_1_1_0 and _0_1_1_0; the names in their containers
        // end _1_1_1_1 and _0_1_1_1.
        for name in ["1_1_1_0", "0_1_1_0", "1_1_1_1", "0_1_1_1"] {
            let container = format!("indirect_diffuse_lighting:CS_FinalGather_{name}_a1d0fa7d_Wrapped");
            assert_eq!(role(container.as_bytes()), Some(Role::Gather));
        }
    }

    #[test]
    fn backlight_forms() {
        assert_eq!(backlight_form(120, 42), Some(Form::Packed));
        assert_eq!(backlight_form(112, 42), Some(Form::Packed));
        assert_eq!(backlight_form(112, 10), Some(Form::Float));
        assert_eq!(backlight_form(112, 2), Some(Form::Float));
        // Only integers where the packed light goes; nothing else anywhere.
        assert_eq!(backlight_form(120, 10), None);
        assert_eq!(backlight_form(112, 28), None);
        assert_eq!(backlight_form(176, 42), None);
    }

    #[test]
    fn backlight_switch() {
        assert!(!kept_by(""));
        assert!(!kept_by("0"));
        assert!(kept_by("1"));
        assert!(kept_by(" on "));
    }

    /// The passes compile with the system's compiler and the host takes them.
    #[test]
    fn the_passes_build() {
        let Some(host) = host() else {
            return;
        };
        make(&host.device, |bytes| host.uav_buffer(bytes)).expect("the passes");
        make_clear(&host.device).expect("the clears");
    }
}
