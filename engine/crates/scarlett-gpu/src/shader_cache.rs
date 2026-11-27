//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! (the BLAKE3 of its sources) and its settings, so a changed translator
//! never serves an old translation. Entries go through
//! `scarlett-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use scarlett_pipeline::{Cache, Converter};
use scarlett_shader::emit::{HostShader, StaticSampler, UserData};
use scarlett_shader::resource::RootIndexedView;
use scarlett_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies a translation beside its container: the translator
/// (the BLAKE3 of its sources, `build.rs`), the settings it runs with, and
/// what it was told of the host. Not our executable: a rebuild of anything
/// else used to translate every shader again.
fn build() -> [u8; 32] {
    // The view heap's null slot differs by host GPU: a translation cached
    // without the clamp must not be served (`host_settings` below).
    crate::descriptor::host::probe_view_limit();
    static FIXED: OnceLock<Vec<u8>> = OnceLock::new();
    let fixed = FIXED.get_or_init(|| {
        let limit = scarlett_shader::emit::loop_limit();
        if limit != scarlett_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "SCARLETT_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let mut bytes = env!("SCARLETT_TRANSLATOR").as_bytes().to_vec();
        bytes.extend_from_slice(&limit.to_le_bytes());
        // Which group barriers wait is part of the translation too.
        bytes.push(scarlett_shader::emit::group_sync());
        // So are the switches that pick the form of a wave64 program:
        // without them here a run with `SCARLETT_WAVE_WIDE=0` was
        // served the translations an earlier run of the same
        // executable had cached with the switch unset.
        for name in [
            "SCARLETT_WAVE_WIDE",
            "SCARLETT_WAVE_DEEP_SKIPS",
            "SCARLETT_WAVE_OWN_BITS",
            "SCARLETT_WAVE_LOCAL",
            "SCARLETT_F16_NEAREST",
            "SCARLETT_HELPER_OWN",
            "SCARLETT_FIRSTLANE_OWN",
            "SCARLETT_FIRST_HALF",
            "SCARLETT_STRICT_JOINS",
            "SCARLETT_MESH_OUTPUT_STRICT",
            "SCARLETT_BVH_FENCES",
            "SCARLETT_BVH_BOUNDS",
            // Whether the ray-tracing node test always misses.
            "SCARLETT_RT_MISS",
            "SCARLETT_LDS_FOLD_STARTS",
            "SCARLETT_THREAD_TICKET",
        ] {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(b'=');
            if let Some(value) = std::env::var_os(name) {
                bytes.extend_from_slice(value.as_encoded_bytes());
            }
            bytes.push(0);
        }
        // So is whether pixel programs keep the console's early depth test
        // (`SCARLETT_EARLY_DEPTH=1`): only while it is on, so that the
        // translations cached without it are still served without it.
        if scarlett_shader::emit::early_depth() {
            tracing::info!(
                "SCARLETT_EARLY_DEPTH=1: pixel programs whose console code tests depth before them get the host's early depth-stencil test"
            );
            bytes.extend_from_slice(b"SCARLETT_EARLY_DEPTH=1");
            bytes.push(0);
        }
        // And which clean-up passes ran over the translated code.
        bytes.extend_from_slice(&scarlett_shader::emit::shader_opt().to_le_bytes());
        bytes
    });
    let mut hash = blake3::Hasher::new();
    hash.update(fixed);
    for value in scarlett_shader::emit::host_settings() {
        hash.update(&value.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

/// What tells one set of translations from another ([`build`]).
pub(crate) fn translation() -> [u8; 32] {
    build()
}

/// Puts the translation of `container` into the cache before any title asks
/// for it (`scarlett shaders`): whether it was there already, or why it does
/// not translate. The host is opened first, as a run does: what the card
/// can do decides the form of some programs.
pub fn precompile(container: &[u8]) -> Result<bool, String> {
    let _ = crate::host::host();
    let misses = MISSES.load(Ordering::Relaxed);
    host_shader(container)?;
    // (Counts other threads' misses too: the caller only adds them up.)
    Ok(MISSES.load(Ordering::Relaxed) == misses)
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let translate = |c: &[u8]| scarlett_shader::emit::translate(c).map_err(|e| e.to_string());
    let ours = match CACHE.get() {
        Some(cache) => cached(cache, &build(), container, translate),
        None => translate(container),
    };
    paired(container, ours)
}

/// Paired wave64 code from another translator (ports branch, after pepezin-ai's
/// `paired_override`, f110b577). `SCARLETT_PAIRED_SHADER_DIR=<dir>` holds `<hash>.vsh`
/// entries in this file's encoding, made by fxceless's translator in its paired form
/// (each host thread runs console lanes n and n+32: PiloteCode's form, fxceless #1397):
/// on 32-lane cards our wave-wide form of the volumetric water light draws dark tiles and
/// a black screen under water, and the hair shadow masks have no 32-lane form here at
/// all. An entry's code replaces ours only for a compute program on a 32-lane host with
/// the views capped at one million and no raw twins (what the entries were made for),
/// and only when everything else it binds equals our own translation: the one served,
/// or, where we have none, the program as a 64-lane host would bind it.
fn paired(container: &[u8], ours: Result<HostShader, String>) -> Result<HostShader, String> {
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;
    static DIR: OnceLock<Option<(PathBuf, HashSet<String>)>> = OnceLock::new();
    let Some((dir, hashes)) = DIR.get_or_init(|| {
        let dir = PathBuf::from(std::env::var_os("SCARLETT_PAIRED_SHADER_DIR")?);
        let hashes: HashSet<String> = std::fs::read_dir(&dir)
            .ok()?
            .filter_map(|e| {
                let name = e.ok()?.file_name().into_string().ok()?;
                let hash = name.strip_suffix(".vsh")?;
                (hash.len() == 16).then(|| hash.to_ascii_lowercase())
            })
            .collect();
        let settings = scarlett_shader::emit::host_settings();
        if settings[..4] != [32, 999_999, 0, 0] {
            tracing::warn!(?settings, "paired shaders: made for a 32-lane host with 1,000,000 views; not used");
            return None;
        }
        tracing::info!(dir = %dir.display(), entries = hashes.len(), "paired shaders");
        Some((dir, hashes))
    }) else {
        return ours;
    };
    let digest = blake3::hash(container).to_hex();
    let hash = &digest.as_str()[..16];
    if !hashes.contains(hash) {
        return ours;
    }
    static DONE: OnceLock<Mutex<HashMap<String, Option<HostShader>>>> = OnceLock::new();
    let done = DONE.get_or_init(Default::default);
    if let Some(known) = done.lock().unwrap_or_else(|e| e.into_inner()).get(hash) {
        return known.clone().ok_or(()).or(ours);
    }
    let spliced = splice(dir, hash, container, ours.as_ref().ok());
    done.lock().unwrap_or_else(|e| e.into_inner()).insert(hash.to_owned(), spliced.clone());
    spliced.ok_or(()).or(ours)
}

fn splice(dir: &std::path::Path, hash: &str, container: &[u8], ours: Option<&HostShader>) -> Option<HostShader> {
    let path = dir.join(format!("{hash}.vsh"));
    let Some(theirs) = std::fs::read(&path).ok().and_then(|b| decode(&b)) else {
        tracing::warn!(hash, path = %path.display(), "paired shader entry does not read");
        return None;
    };
    if theirs.stage != HwStage::Cs || !theirs.dxil.starts_with(b"DXBC") {
        tracing::warn!(hash, stage = ?theirs.stage, "paired shader entry is not a compute program");
        return None;
    }
    // An entry its maker could not pair comes out as a 64-lane program (the
    // decal trail and particle emitters, scar7): a 32-lane card refuses its
    // pipeline, so it is no better than ours.
    if let Some((least, _)) = wave_lanes(&theirs.dxil)
        && least > 32
    {
        tracing::warn!(hash, least, "paired shader entry needs waves of 64 lanes (not paired); not used");
        return None;
    }
    // Everything but the code, against one of our translations.
    let same = |ours: &HostShader| {
        let mut ours = ours.clone();
        ours.dxil.clone_from(&theirs.dxil);
        ours == theirs
    };
    let differs = |ours: &HostShader| {
        let mut d = Vec::new();
        if ours.stage != theirs.stage { d.push("stage"); }
        if ours.user_data != theirs.user_data { d.push("user_data"); }
        if ours.samplers != theirs.samplers { d.push("samplers"); }
        if ours.static_samplers != theirs.static_samplers { d.push("static_samplers"); }
        if ours.root_cbvs != theirs.root_cbvs { d.push("root_cbvs"); }
        if ours.root_srvs != theirs.root_srvs { d.push("root_srvs"); }
        if ours.root_uavs != theirs.root_uavs { d.push("root_uavs"); }
        if ours.root_indexed_views != theirs.root_indexed_views { d.push("root_indexed_views"); }
        d.join(",")
    };
    if let Some(ours) = ours {
        if same(ours) {
            tracing::info!(hash, "paired wave64 code used (bindings equal ours)");
            return Some(theirs);
        }
        tracing::info!(hash, differs = %differs(ours), "paired shader: our form binds differently; trying the 64-lane one");
        tracing::debug!(hash, ours = ?ours.user_data, theirs = ?theirs.user_data, "paired shader bindings");
    }
    match scarlett_shader::emit::with_host_wave(64, || scarlett_shader::emit::translate(container)) {
        Ok(wide) if same(&wide) => {
            tracing::info!(hash, "paired wave64 code used (bindings equal our 64-lane form)");
            Some(theirs)
        }
        Ok(wide) => {
            tracing::warn!(hash, differs = %differs(&wide), "paired shader binds differently from ours; not used");
            tracing::debug!(hash, ours = ?wide.user_data, theirs = ?theirs.user_data, ours_uavs = ?wide.root_uavs, theirs_uavs = ?theirs.root_uavs, "paired shader bindings");
            None
        }
        Err(e) => {
            tracing::warn!(hash, error = %e, "paired shader: no translation of ours to compare with; not used");
            None
        }
    }
}

/// The least and most wave lanes a DXIL container's pipeline state
/// validation part (`PSV0`) asks for, if it says.
fn wave_lanes(container: &[u8]) -> Option<(u32, u32)> {
    let word = |at: usize| Some(u32::from_le_bytes(container.get(at..at + 4)?.try_into().ok()?));
    let parts = word(28)? as usize;
    (0..parts.min(64)).find_map(|n| {
        let at = word(32 + 4 * n)? as usize;
        (container.get(at..at + 4)? == b"PSV0").then_some(())?;
        // the part's size, then the runtime info's size and the info: a
        // 16-byte stage union, then the least and most wave lanes
        let info = word(at + 8)? as usize;
        (info >= 24).then_some(())?;
        Some((word(at + 12 + 16)?, word(at + 12 + 20)?))
    })
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_lanes_read_from_the_psv_part() {
        // header (32 bytes, 2 parts), the offsets, an SFI0 part, a PSV0 part
        let mut c = b"DXBC".to_vec();
        c.extend([0; 16]);
        c.extend(1u32.to_le_bytes());
        c.extend(0u32.to_le_bytes());
        c.extend(2u32.to_le_bytes());
        let first = 40u32;
        let second = first + 16;
        c.extend(first.to_le_bytes());
        c.extend(second.to_le_bytes());
        c.extend(b"SFI0");
        c.extend(8u32.to_le_bytes());
        c.extend(0u64.to_le_bytes());
        c.extend(b"PSV0");
        c.extend(32u32.to_le_bytes());
        c.extend(24u32.to_le_bytes());
        c.extend([0; 16]);
        c.extend(64u32.to_le_bytes());
        c.extend(64u32.to_le_bytes());
        assert_eq!(wave_lanes(&c), Some((64, 64)));
        assert_eq!(wave_lanes(&c[..60]), None);
        assert_eq!(wave_lanes(b"DXBC"), None);
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}

/// The shader translator's fingerprint (a hash of its sources, from build.rs),
/// part of every cache key; `scarlett fingerprint` prints it.
pub const TRANSLATOR: &str = env!("SCARLETT_TRANSLATOR");
