//! Chat runtime shared by `ferrum-cli` and `ferrum-server`: a loaded model,
//! its tokenizer and chat template, one session with prefix reuse, and a
//! streaming split of generated text into reasoning, content and tool calls.
#![forbid(unsafe_code)]
use super::{
    HybridConfig, LoadedHybrid, Variant,
    chat::{ChatTemplate, ParsedOutput, parse_output},
    config::uint,
    draft::{DraftModel, DraftOptions, DraftQuant, TreeSearch, check_drafter_tokenizer},
    engine::ImageSpan,
    engine::{SpecGeometry, spec_bytes},
    lookup::Lookup,
    mtp::{Mtp, mtp_state_bytes, mtp_weight_bytes},
    plan::{DraftMemory, PlanOptions},
    session::{
        Completion, Flow, Generation, GenerationHooks, SamplingParams, SaveStats, Session,
        StopReason,
    },
    speculative::Drafter,
};
use crate::{
    Error, MetalDevice, Result,
    loader::gguf::{GgufFile, MetadataValue},
    tokenizer::Tokenizer,
    vision::{AudioModel, ImageBudget, ImageEmbedding, VisionModel, media},
};
use serde_json::{Map, Value as Json};
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};

/// The placeholder token a chat template emits for one image.
const IMAGE_PAD: &str = "<|image_pad|>";
/// Encoded images kept so conversations do not re-encode their history, in
/// full-size images: smaller ones (a request's lower cap) fit more.
const IMAGE_CACHE: usize = 8;

/// A cached image: its content hash and the (fewest, most) tokens it was
/// encoded with.
type ImageKey = (u64, (usize, usize));

/// The vision projector next to the language model, and recent embeddings
/// (by content and the token limits they were encoded with), shared by every
/// session.
pub struct Vision {
    pub model: VisionModel,
    pub path: PathBuf,
    cache: Vec<(ImageKey, Rc<ImageEmbedding>)>,
    /// Time spent encoding (not cached) images for the latest request.
    pub last_encode: std::time::Duration,
}

/// The speech encoder next to the language model (LFM3-Audio's projector),
/// and recently encoded clips by content, shared by every session.
pub struct Audio {
    pub model: AudioModel,
    pub path: PathBuf,
    cache: Vec<(u64, Rc<ImageEmbedding>)>,
    /// Time spent encoding (not cached) clips for the latest request.
    pub last_encode: std::time::Duration,
}

/// Where speculative drafts come from.
#[derive(Debug, Clone, PartialEq)]
pub enum DraftSource {
    /// The target GGUF's own NextN/MTP block.
    Mtp,
    /// A DFlash / DFlash2 / DSpark checkpoint directory (or HF cache entry),
    /// or a DFlash / DFlash2 GGUF from llama.cpp's converter.
    Checkpoint(PathBuf),
    /// Prompt lookup (`lookup::Lookup`): drafts copied from the context.
    Lookup,
}

impl DraftSource {
    /// `mtp`, `lookup` or a path.
    pub fn parse(value: &str) -> Self {
        if value.eq_ignore_ascii_case("mtp") {
            Self::Mtp
        } else if value.eq_ignore_ascii_case("lookup") {
            Self::Lookup
        } else {
            Self::Checkpoint(value.into())
        }
    }
}

/// Default prompt-lookup drafts per step (`--draft-max` overrides).
const LOOKUP_DRAFTS: usize = 3;
/// Most draft-tree nodes: the verify takes the anchor plus at most 31 rows.
const MAX_TREE_NODES: usize = 31;

/// The KFlash budget's cache key: the model, the drafter and its block
/// depth, and the chip (costs and calibration are each of these).
pub fn tree_rule_key(
    path: &Path,
    drafter: Option<&dyn super::speculative::Drafter>,
    device: &MetalDevice,
) -> String {
    // A checkpoint directory by its name (dots and all), a GGUF by its stem.
    let name = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    };
    let model = name.map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    let drafter = drafter.map_or_else(String::new, |d| format!("{}-k{}", d.name(), d.max_drafts()));
    format!("{model}-{drafter}-{}", device.name())
}

/// Speculative decoding setup for `Runtime::load_speculative`.
#[derive(Debug, Clone)]
pub struct SpecOptions {
    pub source: DraftSource,
    /// Drafts per step (None: 3 for MTP, 7 with best-first trees and 5 with
    /// sibling trees; the trained block for dense
    /// targets and 2 for MoE targets, whose verify cost grows with the
    /// number of distinct experts).
    pub max_drafts: Option<usize>,
    /// Draft trees of at most this many nodes (None:
    /// chains). Drafts per step bound the trees' depth.
    pub tree: Option<usize>,
    /// Let KFlash's budget pick each cycle's tree rows within `tree` and
    /// calibrate the drafter's lattice (`--draft-tree auto`).
    pub tree_rule: bool,
    pub draft: DraftOptions,
}

impl SpecOptions {
    pub fn new(source: DraftSource) -> Self {
        Self {
            source,
            max_drafts: None,
            tree: None,
            tree_rule: false,
            draft: DraftOptions::default(),
        }
    }

    /// Apply one command-line flag shared by `ferrum-cli` and
    /// `ferrum-server`; returns false when `key` is not a draft flag.
    /// `--draft mtp|lookup|DIR|FILE.gguf` enables speculation; the others tune it:
    /// `--draft-max N`, `--draft-tree N|auto` (draft trees of N nodes for
    /// requests; auto: as many rows as pay each cycle, learned), `--draft-tree-search best-first|siblings` (how MTP grows
    /// them), `--draft-quant q8_0|q4_0`, `--draft-context N`
    /// (context slots for drafts without a sliding window) and
    /// `--draft-p-min P` (DSpark confidence cut-off).
    pub fn apply_flag(spec: &mut Option<Self>, key: &str, value: &str) -> Result<bool> {
        let bad = || Error::Parameter(format!("invalid {key}: {value}"));
        if key == "--draft" {
            let previous = spec.take();
            let mut next = Self::new(DraftSource::parse(value));
            if let Some(p) = previous {
                next.max_drafts = p.max_drafts;
                next.tree = p.tree;
                next.tree_rule = p.tree_rule;
                next.draft = p.draft;
            }
            *spec = Some(next);
            return Ok(true);
        }
        if !key.starts_with("--draft-") {
            return Ok(false);
        }
        let s = spec.get_or_insert_with(|| Self::new(DraftSource::Mtp));
        match key {
            "--draft-max" => s.max_drafts = Some(value.parse().map_err(|_| bad())?),
            "--draft-tree" if value == "auto" => {
                s.tree = Some(MAX_TREE_NODES);
                s.tree_rule = true;
                s.draft.kflash = true;
            }
            "--draft-tree" => {
                let budget: usize = value.parse().map_err(|_| bad())?;
                if !(1..=MAX_TREE_NODES).contains(&budget) {
                    return Err(Error::Parameter(format!(
                        "--draft-tree takes 1..={MAX_TREE_NODES} nodes"
                    )));
                }
                s.tree = Some(budget);
            }
            "--draft-quant" => {
                s.draft.quant = match value.to_ascii_lowercase().as_str() {
                    "q8_0" => DraftQuant::Q8_0,
                    "q4_0" => DraftQuant::Q4_0,
                    _ => return Err(bad()),
                }
            }
            "--draft-context" => s.draft.context_cap = value.parse().map_err(|_| bad())?,
            "--draft-p-min" => s.draft.confidence_min = value.parse().map_err(|_| bad())?,
            "--draft-tree-search" => {
                s.draft.tree_search = match value {
                    "best-first" => TreeSearch::BestFirst,
                    "siblings" => TreeSearch::Siblings,
                    _ => return Err(bad()),
                }
            }
            "--draft-vocab" => s.draft.vocab = Some(value.parse().map_err(|_| bad())?),
            "--draft-block" => s.draft.block = Some(value.parse().map_err(|_| bad())?),
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn drafts(&self, target: &HybridConfig, trained: usize) -> usize {
        self.max_drafts
            .unwrap_or(match (&self.source, target.variant) {
                // Measured best: best-first trees 7 deep, sibling trees 5, chains 3.
                (DraftSource::Mtp, _) if self.tree.is_some() => match self.draft.tree_search {
                    TreeSearch::BestFirst => 7,
                    TreeSearch::Siblings => 5,
                },
                // On MoE targets each draft routes to its own experts: 2 beats 3
                // (docs/measurements/mtp-moe).
                (DraftSource::Mtp, Variant::Moe) => 2,
                (DraftSource::Mtp, _) => 3,
                (DraftSource::Lookup, _) => LOOKUP_DRAFTS,
                (_, Variant::Moe | Variant::Flint) => trained.min(2),
                _ => trained,
            })
    }

    /// Verify rows for `drafts` per step: the anchor plus the drafts, or the
    /// tree's nodes if wider.
    fn verify_rows(&self, drafts: usize) -> usize {
        drafts.max(self.tree.unwrap_or(0)) + 1
    }

    /// Memory the drafter will take next to `target`, before loading. `file`
    /// is the target's GGUF (None for an MLX checkpoint).
    pub fn memory(
        &self,
        file: Option<&GgufFile>,
        target: &HybridConfig,
        chunk: usize,
    ) -> Result<DraftMemory> {
        match &self.source {
            DraftSource::Mtp => {
                let file = file.ok_or_else(|| {
                    Error::Config("the MTP drafter reads the NextN block of a GGUF target".into())
                })?;
                let k = self.drafts(target, 3);
                let geometry = SpecGeometry {
                    rows: self.verify_rows(k),
                    aux_layers: Vec::new(),
                    final_hidden: true,
                };
                let per_token =
                    (2 * target.kv_cache.row_bytes(target.kv_heads * target.head_dim)) as f64;
                Ok(DraftMemory {
                    weights: mtp_weight_bytes(file, target),
                    fixed: mtp_state_bytes(target, 0, chunk, k, self.tree.unwrap_or(0))
                        + spec_bytes(target, chunk, &geometry),
                    per_token,
                })
            }
            DraftSource::Lookup => {
                let geometry = SpecGeometry {
                    rows: self.verify_rows(self.drafts(target, LOOKUP_DRAFTS)),
                    aux_layers: Vec::new(),
                    final_hidden: false,
                };
                Ok(DraftMemory {
                    weights: 0,
                    fixed: spec_bytes(target, chunk, &geometry),
                    per_token: 0.,
                })
            }
            DraftSource::Checkpoint(dir) => {
                // An MLX target's tokenizer is its own tokenizer.json; the
                // drafter's config still checks the vocabulary below.
                if let Some(file) = file {
                    check_drafter_tokenizer(dir, file.metadata())?;
                }
                let (config, weights, state) = DraftModel::memory(dir, target, chunk, self.draft)?;
                let k = self.drafts(target, config.max_drafts());
                let geometry = SpecGeometry {
                    rows: self.verify_rows(k),
                    aux_layers: config.target_layers.clone(),
                    final_hidden: false,
                };
                Ok(DraftMemory {
                    weights,
                    fixed: state + spec_bytes(target, chunk, &geometry),
                    per_token: 0.,
                })
            }
        }
    }

    /// Load the drafter for a session of `capacity` positions.
    pub fn build(
        &self,
        d: &MetalDevice,
        target_path: &Path,
        loaded: &LoadedHybrid,
        capacity: usize,
    ) -> Result<Box<dyn Drafter>> {
        let model = &loaded.model;
        Ok(match &self.source {
            DraftSource::Mtp => {
                let k = self.drafts(&model.config, 3);
                Box::new(Mtp::load(
                    d,
                    target_path,
                    model,
                    capacity,
                    k,
                    self.draft.vocab,
                    self.tree.unwrap_or(0),
                    self.draft.tree_search,
                )?)
            }
            DraftSource::Lookup => Box::new(Lookup::new(self.drafts(&model.config, LOOKUP_DRAFTS))),
            DraftSource::Checkpoint(dir) => {
                if !super::mlx::is_mlx_dir(target_path) {
                    let header = std::fs::File::open(target_path)
                        .map_err(|e| Error::Gguf(format!("{}: {e}", target_path.display())))?;
                    let header = GgufFile::from_reader_skipping_arrays(header, 4096)?;
                    check_drafter_tokenizer(dir, header.metadata())?;
                }
                let (config, _, _) =
                    DraftModel::memory(dir, &model.config, model.chunk(), self.draft)?;
                let k = self.drafts(&model.config, config.max_drafts());
                Box::new(DraftModel::load(
                    d,
                    dir,
                    model,
                    capacity,
                    Some(k),
                    self.draft,
                )?)
            }
        })
    }
}

/// Incremental text produced while generating.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Json>,
    pub tools: Vec<Json>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Extra chat-template variables (`enable_thinking`, `reasoning_effort`, ...).
    pub template_vars: Map<String, Json>,
    /// Stop when any of these strings appears in the visible content.
    pub stop: Vec<String>,
    /// Most reasoning tokens before `</think>` is forced (None = unlimited).
    pub reasoning_budget: Option<usize>,
    /// Leave reasoning inline (with its tags) in the content instead of
    /// splitting it out (llama-server `--reasoning-format none`).
    pub raw_reasoning: bool,
    /// Token limits for this request's images; an image part's own
    /// `detail`, `min_tokens` or `max_tokens` take precedence.
    pub image_budget: ImageBudget,
}

impl ChatRequest {
    pub fn new(messages: Vec<Json>, sampling: SamplingParams) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            max_tokens: usize::MAX,
            sampling,
            template_vars: Map::new(),
            stop: Vec::new(),
            reasoning_budget: None,
            raw_reasoning: false,
            image_budget: ImageBudget::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub output: ParsedOutput,
    pub completion: Completion,
    /// True when a `stop` string ended generation.
    pub stopped_by_string: bool,
    /// Generated tokens that belonged to the reasoning block.
    pub reasoning_tokens: usize,
    /// True when the thinking budget closed the reasoning block.
    pub budget_exhausted: bool,
    /// Raw generated text (reasoning, content and tool-call markup).
    pub text: String,
}

/// Callbacks for one chat or completion request.
pub trait ChatHooks {
    fn delta(&mut self, delta: Delta) -> Result<()>;
    /// Prompt length and tokens reused from the session, before prefill.
    fn begin(&mut self, _prompt: usize, _reused: usize) {}
    /// Prefill progress over the new prompt tokens; false cancels.
    fn prefill(&mut self, _processed: usize, _total: usize) -> bool {
        true
    }
    /// Polled after every token; true stops generation.
    fn cancelled(&self) -> bool {
        false
    }
    /// Called after every generated token with the running count.
    fn generated(&mut self, _tokens: usize) {}
}

impl<F: FnMut(Delta) -> Result<()>> ChatHooks for F {
    fn delta(&mut self, delta: Delta) -> Result<()> {
        self(delta)
    }
}

/// A chat turn or completion in progress on one session (`Runtime::start_chat`).
pub struct Turn {
    generation: Generation,
    driver: Driver,
    chat: Option<ChatTurn>,
}

struct ChatTurn {
    starts_in_reasoning: bool,
    raw_reasoning: bool,
    tools: Vec<Json>,
}

impl Turn {
    /// Prompt length and tokens reused from the session's cache.
    pub fn prompt(&self) -> (usize, usize) {
        self.generation.prompt()
    }
    /// Past the prompt: generating tokens.
    pub fn decoding(&self) -> bool {
        self.generation.decoding()
    }
}

/// A rendered chat prompt (`Runtime::chat_prompt`).
pub struct ChatPrompt {
    pub tokens: Vec<u32>,
    /// Where its images sit in `tokens`.
    pub spans: Vec<ImageSpan>,
    pub text: String,
    /// Where the reply's header starts, and whether the prompt ends in a
    /// user message (`Session::set_checkpoint`).
    pub checkpoint: Option<(usize, bool)>,
}

/// What a finished `Turn` produced.
pub enum Done {
    Chat(Box<ChatResult>),
    /// A raw completion: its text (cut at a stop string) and whether one hit.
    Completion(Completion, String, bool),
}

pub struct Runtime {
    pub device: MetalDevice,
    pub loaded: LoadedHybrid,
    pub template: ChatTemplate,
    /// Slot 0, the one `chat` and `complete` use.
    pub session: Session,
    /// Slots 1.. (`PlanOptions::parallel`): sessions of their own sharing the
    /// model, drafter-free. The server interleaves requests across them.
    pub slots: Vec<Session>,
    /// The model's recommended sampling (GGUF `general.sampling.*`).
    pub default_sampling: SamplingParams,
    pub model_name: String,
    /// Device bytes the drafter took (weights, buffers, verify buffers).
    pub draft_loaded_bytes: usize,
    /// Present when a vision projector was loaded: requests may carry images.
    pub vision: Option<Vision>,
    /// Present when a speech projector was loaded: requests may carry audio.
    pub audio: Option<Audio>,
    /// The model file as a saved session depends on it (`save_session`).
    cache_identity: String,
}

impl Runtime {
    /// Compile and page in every kernel path (chunked prefill, single-token
    /// decode, sampling candidates) so the first real request runs at full
    /// speed, then clear the session.
    pub fn warmup(&mut self) -> Result<()> {
        let model = &self.loaded.model;
        let n = 66.min(model.chunk() + 2).min(self.session.capacity() - 3);
        let prompt: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 20000).collect();
        let result = self.session.generate(
            &self.device,
            model,
            &prompt,
            2,
            &self.default_sampling,
            &[],
            &mut |_| Ok(true),
        );
        self.session.reset();
        result.map(drop)
    }

    pub fn load(path: impl AsRef<Path>, options: PlanOptions) -> Result<Self> {
        Self::load_speculative(path, options, None)
    }

    /// Load with an optional drafter: its memory is planned before any
    /// weight is read (the auto-fitted context shrinks to make room), and the
    /// session speculates by default.
    pub fn load_speculative(
        path: impl AsRef<Path>,
        options: PlanOptions,
        spec: Option<&SpecOptions>,
    ) -> Result<Self> {
        Self::load_multimodal(path, options, spec, None, crate::vision::DEFAULT_MAX_TOKENS)
    }

    /// Like `load_speculative`, additionally loading the vision projector
    /// `mmproj` (images then take at most `image_tokens` positions each). Its
    /// memory is planned with the model so the context shrinks to fit it.
    pub fn load_multimodal(
        path: impl AsRef<Path>,
        mut options: PlanOptions,
        spec: Option<&SpecOptions>,
        mmproj: Option<&Path>,
        image_tokens: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        if let Some(mmproj) = mmproj {
            options.vision = crate::vision::projector_bytes(mmproj);
        }
        let device = MetalDevice::new()?;
        let default_sampling = recommended_sampling_for(path)?;
        if let Some(spec) = spec {
            if super::mlx::is_mlx_dir(path) {
                let mut config = super::mlx::config(path)?;
                config.kv_cache = options.kv_cache;
                options.draft = spec.memory(None, &config, options.chunk.max(1))?;
            } else {
                let file = GgufFile::open(path)?;
                let mut config = HybridConfig::from_gguf(&file)?;
                config.kv_cache = options.kv_cache;
                options.draft = spec.memory(Some(&file), &config, options.chunk.max(1))?;
            }
        }
        let loaded = super::load_with(&device, path, options)?;
        let (bos, eos) = &loaded.chat_special_tokens;
        let template = ChatTemplate::with_special_tokens(
            loaded
                .chat_template
                .as_deref()
                .ok_or_else(|| Error::Tokenizer("GGUF has no chat template".into()))?,
            bos,
            eos,
        )?
        .with_opens_thinking(loaded.model.config.variant == Variant::Flint);
        let mut session = Session::new(
            &device,
            &loaded.model,
            loaded.plan.context,
            options.snapshots,
        )?;
        let slots = (1..options.parallel.max(1))
            .map(|_| {
                Session::new(
                    &device,
                    &loaded.model,
                    loaded.plan.context,
                    options.snapshots,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut draft_loaded_bytes = 0;
        if let Some(spec) = spec {
            let before = device.allocated_bytes();
            let drafter = spec.build(&device, path, &loaded, loaded.plan.context)?;
            session.set_drafter(&device, &loaded.model, drafter)?;
            if spec.tree.is_some() {
                session.set_tree_budget(&device, &loaded.model, spec.tree)?;
            }
            if spec.tree_rule {
                let key = tree_rule_key(path, session.drafter(), &device);
                session.set_tree_rule(true, Some(&key));
            }
            draft_loaded_bytes = device.allocated_bytes().saturating_sub(before);
        }
        let audio_projector = mmproj.filter(|p| crate::vision::is_audio_projector(p));
        let audio = match audio_projector {
            Some(mmproj) => {
                let model = AudioModel::load(&device, mmproj)?;
                if model.out_dim() != loaded.model.config.hidden {
                    return Err(Error::Config(format!(
                        "{} projects to width {} but the language model is {} wide",
                        mmproj.display(),
                        model.out_dim(),
                        loaded.model.config.hidden
                    )));
                }
                model.warm(&device)?;
                Some(Audio {
                    model,
                    path: mmproj.to_owned(),
                    cache: Vec::new(),
                    last_encode: std::time::Duration::ZERO,
                })
            }
            None => None,
        };
        let vision = match mmproj.filter(|_| audio_projector.is_none()) {
            Some(mmproj) => {
                let model = VisionModel::load(&device, mmproj, image_tokens)?;
                if model.config.out_dim != loaded.model.config.hidden {
                    return Err(Error::Config(format!(
                        "{} projects to width {} but the language model is {} wide",
                        mmproj.display(),
                        model.config.out_dim,
                        loaded.model.config.hidden
                    )));
                }
                // Qwen places image tokens on a 2-D grid and needs the section
                // sizes. LFM3 gives each image token the next position.
                if !model.linear_positions() && loaded.model.config.rope_sections[1] == 0 {
                    return Err(Error::Config(
                        "the language model declares no multi-axis RoPE sections, so it cannot place images"
                            .into(),
                    ));
                }
                model.warm(&device)?;
                Some(Vision {
                    model,
                    path: mmproj.to_owned(),
                    cache: Vec::new(),
                    last_encode: std::time::Duration::ZERO,
                })
            }
            None => None,
        };
        let model_name = path.file_stem().map_or_else(
            || loaded.model.config.name.clone(),
            |s| s.to_string_lossy().into_owned(),
        );
        Ok(Self {
            device,
            loaded,
            template,
            session,
            slots,
            default_sampling,
            model_name,
            draft_loaded_bytes,
            vision,
            audio,
            // Cached image rows depend on the projector as well.
            cache_identity: match mmproj {
                Some(mmproj) => format!("{} + {}", cache_identity(path), cache_identity(mmproj)),
                None => cache_identity(path),
            },
        })
    }

    /// Save slot 0's session to `dir` (see `session::Session::save_to`).
    pub fn save_session(&self, dir: &Path) -> Result<SaveStats> {
        self.session.save_to(dir, &self.cache_identity)
    }

    /// Load slot 0's session from `dir` if it was saved for this model file
    /// and session geometry; returns the tokens now cached.
    pub fn load_session(&mut self, dir: &Path) -> Result<Option<usize>> {
        let model = &self.loaded.model;
        self.session
            .load_from(&self.device, model, dir, &self.cache_identity)
    }

    pub fn supports_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn supports_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Sessions (slot 0 plus `slots`).
    pub fn parallel(&self) -> usize {
        1 + self.slots.len()
    }

    /// Session `slot` (0 is `session`).
    pub fn slot(&self, slot: usize) -> &Session {
        match slot {
            0 => &self.session,
            i => &self.slots[i - 1],
        }
    }

    pub fn slot_mut(&mut self, slot: usize) -> &mut Session {
        match slot {
            0 => &mut self.session,
            i => &mut self.slots[i - 1],
        }
    }

    /// Encode `images` (cached by content and size), failing when no
    /// projector is loaded. `budget` fills what each image leaves unset.
    fn embed_images(
        &mut self,
        images: &[media::ImageInput],
        budget: ImageBudget,
    ) -> Result<Vec<Rc<ImageEmbedding>>> {
        if let Some(vision) = self.vision.as_mut() {
            vision.last_encode = std::time::Duration::ZERO;
        }
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let Some(vision) = self.vision.as_mut() else {
            return Err(Error::Parameter(
                "this model was loaded without a vision projector, so it cannot read images (load an mmproj file with --mmproj)"
                    .into(),
            ));
        };
        let mut out = Vec::with_capacity(images.len());
        for image in images {
            let budget = image.budget.or(budget);
            let key = (
                crate::vision::content_hash(&image.bytes),
                vision.model.limits(budget),
            );
            if let Some(i) = vision.cache.iter().position(|(k, _)| *k == key) {
                let hit = vision.cache.remove(i);
                out.push(hit.1.clone());
                vision.cache.push(hit);
                continue;
            }
            let start = std::time::Instant::now();
            let embedding = Rc::new(vision.model.encode_with(
                &self.device,
                &image.bytes,
                budget,
            )?);
            vision.last_encode += start.elapsed();
            vision.cache.push((key, embedding.clone()));
            // Oldest first, until the rest fit; the newest always stays.
            let cap = IMAGE_CACHE * vision.model.max_tokens;
            let mut held: usize = vision.cache.iter().map(|(_, e)| e.tokens()).sum();
            while held > cap && vision.cache.len() > 1 {
                held -= vision.cache.remove(0).1.tokens();
            }
            out.push(embedding);
        }
        Ok(out)
    }

    /// Encode WAV `clips` (cached by content), failing when no speech
    /// projector is loaded.
    fn embed_audio(&mut self, clips: &[Vec<u8>]) -> Result<Vec<Rc<ImageEmbedding>>> {
        if let Some(audio) = self.audio.as_mut() {
            audio.last_encode = std::time::Duration::ZERO;
        }
        if clips.is_empty() {
            return Ok(Vec::new());
        }
        let Some(audio) = self.audio.as_mut() else {
            return Err(Error::Parameter(
                "this model was loaded without an audio projector, so it cannot hear audio (load LFM3-Audio's mmproj file with --mmproj)"
                    .into(),
            ));
        };
        let mut out = Vec::with_capacity(clips.len());
        for clip in clips {
            let key = crate::vision::content_hash(clip);
            if let Some(i) = audio.cache.iter().position(|(k, _)| *k == key) {
                let hit = audio.cache.remove(i);
                out.push(hit.1.clone());
                audio.cache.push(hit);
                continue;
            }
            let start = std::time::Instant::now();
            let embedding = Rc::new(audio.model.encode(&self.device, clip)?);
            audio.last_encode += start.elapsed();
            audio.cache.push((key, embedding.clone()));
            if audio.cache.len() > crate::vision::AUDIO_CACHE_CLIPS {
                audio.cache.remove(0);
            }
            out.push(embedding);
        }
        Ok(out)
    }

    /// Replace each image and audio placeholder in `prompt` with one pad
    /// token per position, tokenize, and locate every image and clip in the
    /// result. Clips take one position per row, like LFM3 images.
    fn expand_media(
        &self,
        prompt: &str,
        images: &[Rc<ImageEmbedding>],
        clips: &[Rc<ImageEmbedding>],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>)> {
        let tokenizer = &self.loaded.tokenizer;
        if self.loaded.model.config.variant == Variant::Lfm3 && prompt.contains("<|video|>") {
            return Err(Error::Parameter(
                "video is not supported; LFM3-VL reads still images only".into(),
            ));
        }
        if images.is_empty() && clips.is_empty() {
            return Ok((tokenizer.encode(prompt)?, Vec::new()));
        }
        let image_marker = self
            .vision
            .as_ref()
            .map(|v| v.model.marker())
            .unwrap_or(IMAGE_PAD);
        let linear = self
            .vision
            .as_ref()
            .is_some_and(|v| v.model.linear_positions());
        let audio_marker = self.audio.as_ref().map(|a| a.model.marker());
        let kinds = [(image_marker, images), (audio_marker.unwrap_or(""), clips)];
        let lengths = kinds.map(|(_, items)| items.iter().map(|e| e.tokens()).collect::<Vec<_>>());
        let markers = kinds.map(|(marker, _)| marker);
        let text = splice_media(prompt, markers, [&lengths[0], &lengths[1]])?;
        let tokens = tokenizer.encode(&text)?;
        let pad_of = |marker: &str| -> Result<u32> {
            let pad = tokenizer.encode(marker)?;
            match pad[..] {
                [pad] => Ok(pad),
                _ => Err(Error::Tokenizer(format!("{marker} is not a single token"))),
            }
        };
        let mut pads = [None; 2];
        for (kind, (marker, items)) in kinds.iter().enumerate() {
            if !items.is_empty() {
                pads[kind] = Some(pad_of(marker)?);
            }
        }
        let runs = locate_media(&tokens, pads, [&lengths[0], &lengths[1]])?;
        let spans = runs
            .into_iter()
            .map(|(start, kind, index)| {
                let item = &kinds[kind].1[index];
                ImageSpan {
                    start,
                    grid: item.grid,
                    embedding: item.rows.clone(),
                    deepstack: item.deepstack.clone(),
                    hash: item.hash,
                    linear: linear || kind == 1,
                }
            })
            .collect();
        Ok((tokens, spans))
    }

    pub fn render(&self, request: &ChatRequest) -> Result<String> {
        let (messages, _) = media::take_images(&request.messages)?;
        let (messages, _) = media::take_audio(&messages)?;
        self.render_messages(&messages, request)
    }

    fn render_messages(&self, messages: &[Json], request: &ChatRequest) -> Result<String> {
        let mut vars = request.template_vars.clone();
        normalize_thinking(self.template.source(), &mut vars);
        let mut prompt = self.template.render(
            messages,
            (!request.tools.is_empty()).then_some(&request.tools[..]),
            true,
            &vars,
        )?;
        prompt.push_str(generation_prefix(&self.template, &vars));
        Ok(prompt)
    }

    /// Run one chat turn, reusing whatever prefix the session already holds.
    pub fn chat(&mut self, request: &ChatRequest, mut hooks: impl ChatHooks) -> Result<ChatResult> {
        let prompt = self.chat_prompt(request)?;
        let mut turn = self.start_chat(0, request, prompt, &mut hooks)?;
        loop {
            if let Some(done) = self.advance(0, &mut turn, &mut hooks)? {
                let Done::Chat(result) = done else {
                    unreachable!("a chat turn ends in a chat result")
                };
                return Ok(*result);
            }
        }
    }

    /// Raw text completion (no chat template, no tool parsing).
    pub fn complete(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingParams,
        stop: &[String],
        mut hooks: impl ChatHooks,
    ) -> Result<(Completion, String, bool)> {
        let mut turn = self.start_complete(0, prompt, max_tokens, sampling, stop, &mut hooks)?;
        loop {
            if let Some(done) = self.advance(0, &mut turn, &mut hooks)? {
                let Done::Completion(completion, text, stopped) = done else {
                    unreachable!("a completion ends in a completion")
                };
                return Ok((completion, text, stopped));
            }
        }
    }

    /// Start a chat turn on session `slot` without running it, from the
    /// request's `chat_prompt`, reusing the session's cached prefix.
    /// `advance` runs it a step at a time.
    pub fn start_chat(
        &mut self,
        slot: usize,
        request: &ChatRequest,
        prompt: ChatPrompt,
        hooks: &mut impl ChatHooks,
    ) -> Result<Turn> {
        let starts_in_reasoning = prompt.text.trim_end().ends_with("<think>");
        if let Some((at, pin)) = prompt.checkpoint {
            self.slot_mut(slot).set_checkpoint(at, pin);
        }
        let mut turn = self.start(
            slot,
            &prompt.tokens,
            prompt.spans,
            request.max_tokens,
            &request.sampling,
            &request.stop,
            starts_in_reasoning && !request.raw_reasoning,
            if starts_in_reasoning {
                request.reasoning_budget
            } else {
                None
            },
            true,
            hooks,
        )?;
        turn.chat = Some(ChatTurn {
            starts_in_reasoning,
            raw_reasoning: request.raw_reasoning,
            tools: request.tools.clone(),
        });
        Ok(turn)
    }

    /// Encode a chat request's images and render and tokenize its prompt:
    /// what `start_chat` feeds a session (so a caller can first pick the
    /// session whose cache shares the most of it).
    pub fn chat_prompt(&mut self, request: &ChatRequest) -> Result<ChatPrompt> {
        let (messages, encoded) = media::take_images(&request.messages)?;
        let (messages, clips) = media::take_audio(&messages)?;
        let images = self.embed_images(&encoded, request.image_budget)?;
        let clips = self.embed_audio(&clips)?;
        let text = self.render_messages(&messages, request)?;
        let (tokens, spans) = self.expand_media(&text, &images, &clips)?;
        let checkpoint = self
            .reply_start(&messages, request, &text, &tokens)
            .map(|at| {
                let last = messages.last().and_then(|m| m["role"].as_str());
                (at, last == Some("user"))
            });
        Ok(ChatPrompt {
            tokens,
            spans,
            text,
            checkpoint,
        })
    }

    /// The token position where the generation prompt (the reply's header)
    /// begins in `tokens`, the rendered and tokenized `messages`: the same
    /// render without it is a prefix of `text`, and the rest tokenizes on its
    /// own to the tail of `tokens` (it opens with a special token, so the
    /// split is clean). None when the template does not render that way.
    fn reply_start(
        &self,
        messages: &[Json],
        request: &ChatRequest,
        text: &str,
        tokens: &[u32],
    ) -> Option<usize> {
        let mut vars = request.template_vars.clone();
        normalize_thinking(self.template.source(), &mut vars);
        let tools = (!request.tools.is_empty()).then_some(&request.tools[..]);
        let head = self.template.render(messages, tools, false, &vars).ok()?;
        let tail = text.strip_prefix(head.as_str())?;
        let tail = self.loaded.tokenizer.encode(tail).ok()?;
        (!tail.is_empty() && tokens.ends_with(&tail)).then(|| tokens.len() - tail.len())
    }

    /// `start_chat` for a raw text completion.
    pub fn start_complete(
        &mut self,
        slot: usize,
        prompt: &str,
        max_tokens: usize,
        sampling: &SamplingParams,
        stop: &[String],
        hooks: &mut impl ChatHooks,
    ) -> Result<Turn> {
        if let Some(vision) = self.vision.as_mut() {
            vision.last_encode = std::time::Duration::ZERO;
        }
        if let Some(audio) = self.audio.as_mut() {
            audio.last_encode = std::time::Duration::ZERO;
        }
        let tokens = self.loaded.tokenizer.encode(prompt)?;
        self.start(
            slot,
            &tokens,
            Vec::new(),
            max_tokens,
            sampling,
            stop,
            false,
            None,
            false,
            hooks,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start(
        &mut self,
        slot: usize,
        prompt: &[u32],
        spans: Vec<ImageSpan>,
        max_tokens: usize,
        sampling: &SamplingParams,
        stop: &[String],
        split_reasoning: bool,
        reasoning_budget: Option<usize>,
        tools: bool,
        hooks: &mut impl ChatHooks,
    ) -> Result<Turn> {
        let close_text = reasoning_close(self.loaded.model.config.variant);
        let mut driver = Driver {
            stream: TextStream::default(),
            splitter: Splitter::new(split_reasoning, stop, tools),
            failure: None,
            reasoning_tokens: 0,
            budget: reasoning_budget,
            budget_exhausted: false,
            close: self.loaded.tokenizer.encode(close_text)?,
            close_text,
            count: 0,
        };
        let session = match slot {
            0 => &mut self.session,
            i => &mut self.slots[i - 1],
        };
        session.set_media(spans);
        let generation = session.begin(
            &self.device,
            &self.loaded.model,
            prompt,
            max_tokens,
            sampling,
            &self.loaded.eos_ids,
            &mut Hooked {
                driver: &mut driver,
                hooks,
                tokenizer: &self.loaded.tokenizer,
            },
        )?;
        Ok(Turn {
            generation,
            driver,
            chat: None,
        })
    }

    /// Run the next prompt chunk or decode step of `turn` on session `slot`
    /// (the one it started on); what it produced once it has finished.
    pub fn advance(
        &mut self,
        slot: usize,
        turn: &mut Turn,
        hooks: &mut impl ChatHooks,
    ) -> Result<Option<Done>> {
        let session = match slot {
            0 => &mut self.session,
            i => &mut self.slots[i - 1],
        };
        let mut hooked = Hooked {
            driver: &mut turn.driver,
            hooks,
            tokenizer: &self.loaded.tokenizer,
        };
        let Some(completion) = session.advance(
            &self.device,
            &self.loaded.model,
            &mut turn.generation,
            &mut hooked,
        )?
        else {
            return Ok(None);
        };
        let driver = &mut turn.driver;
        if let Some(e) = driver.failure.take() {
            return Err(e);
        }
        for delta in driver.splitter.finish() {
            hooks.delta(delta)?;
        }
        let stopped = driver.splitter.stopped;
        let visible = driver.splitter.visible_content();
        let text = std::mem::take(&mut driver.splitter.text);
        let Some(chat) = turn.chat.as_ref() else {
            let text = if stopped { visible } else { text };
            return Ok(Some(Done::Completion(completion, text, stopped)));
        };
        let starts_in_reasoning = chat.starts_in_reasoning;
        let mut output = if chat.raw_reasoning {
            let mut parsed = parse_output(&text, false, &chat.tools);
            let prefix = match (starts_in_reasoning, self.template.opens_thinking()) {
                (false, _) => "",
                (true, true) => "<think>",
                (true, false) => "<think>\n",
            };
            if !parsed.tool_calls.is_empty() {
                parsed.content = format!("{prefix}{}", parsed.content);
            } else {
                parsed.content = format!("{prefix}{}", text.trim_end());
            }
            parsed
        } else {
            parse_output(&text, starts_in_reasoning, &chat.tools)
        };
        if stopped {
            output.content = visible;
        }
        Ok(Some(Done::Chat(Box::new(ChatResult {
            stopped_by_string: stopped,
            output,
            completion,
            reasoning_tokens: driver.reasoning_tokens,
            budget_exhausted: driver.budget_exhausted,
            text,
        }))))
    }
}

/// Incremental detokenization, as `tokenizers`' `DecodeStream` does it, but
/// owning no borrow of the tokenizer so a `Turn` can outlive the call.
#[derive(Default)]
struct TextStream {
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl TextStream {
    fn step(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<Option<String>> {
        self.ids.push(id);
        let text = tokenizer.decode_lenient(&self.ids)?;
        if text.len() <= self.prefix.len() || text.ends_with('\u{FFFD}') {
            return Ok(None);
        }
        let Some(new) = text.strip_prefix(self.prefix.as_str()) else {
            return Err(Error::Tokenizer(
                "streamed text does not extend its prefix".into(),
            ));
        };
        let new = new.to_owned();
        let next_index = self.ids.len() - self.prefix_index;
        self.ids.drain(..self.prefix_index);
        self.prefix = tokenizer.decode_lenient(&self.ids)?;
        self.prefix_index = next_index;
        Ok(Some(new))
    }
}

/// Per-turn state that turns sampled tokens into deltas.
struct Driver {
    stream: TextStream,
    splitter: Splitter,
    failure: Option<Error>,
    reasoning_tokens: usize,
    budget: Option<usize>,
    budget_exhausted: bool,
    close: Vec<u32>,
    close_text: &'static str,
    count: usize,
}

/// A `Driver` with the caller's hooks for one `Session::begin`/`advance` call.
struct Hooked<'a, H: ChatHooks> {
    driver: &'a mut Driver,
    hooks: &'a mut H,
    tokenizer: &'a Tokenizer,
}

impl<H: ChatHooks> Hooked<'_, H> {
    fn emit(&mut self, piece: &str) -> bool {
        for delta in self.driver.splitter.push(piece) {
            if let Err(e) = self.hooks.delta(delta) {
                self.driver.failure = Some(e);
                return false;
            }
        }
        true
    }
}

impl<H: ChatHooks> GenerationHooks for Hooked<'_, H> {
    fn begin(&mut self, prompt: usize, reused: usize) {
        self.hooks.begin(prompt, reused);
    }
    fn prefill(&mut self, processed: usize, total: usize) -> bool {
        self.hooks.prefill(processed, total) && !self.hooks.cancelled()
    }
    fn token(&mut self, token: u32) -> Result<Flow> {
        self.driver.count += 1;
        self.hooks.generated(self.driver.count);
        if self.driver.splitter.in_reasoning {
            self.driver.reasoning_tokens += 1;
        }
        let piece = self.driver.stream.step(self.tokenizer, token)?;
        if let Some(piece) = piece
            && !self.emit(&piece)
        {
            return Ok(Flow::Stop);
        }
        if self.driver.splitter.stopped || self.hooks.cancelled() {
            return Ok(Flow::Stop);
        }
        if self.driver.splitter.in_reasoning
            && self
                .driver
                .budget
                .is_some_and(|b| self.driver.reasoning_tokens >= b)
        {
            self.driver.budget_exhausted = true;
            let text = self.driver.close_text;
            if !self.emit(text) {
                return Ok(Flow::Stop);
            }
            return Ok(Flow::Inject(self.driver.close.clone()));
        }
        Ok(Flow::Continue)
    }
}

impl StopReason {
    /// OpenAI `finish_reason` for a completion that produced no tool call.
    pub fn finish_reason(self) -> &'static str {
        match self {
            StopReason::Eos | StopReason::Stopped | StopReason::Cancelled => "stop",
            StopReason::Length | StopReason::ContextFull => "length",
        }
    }
}

/// A model file's path, size and modification time: a saved session is only
/// loaded into the same file.
fn cache_identity(path: &Path) -> String {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let meta = std::fs::metadata(&path).ok();
    let modified = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |t| t.as_secs());
    format!(
        "{} {} {modified}",
        path.display(),
        meta.map_or(0, |m| m.len())
    )
}

/// Text the generation prompt needs when the template does not open the think
/// block itself (Flint). Thinking is on unless the request turns it off. On:
/// open a think block, so the reasoning is split from the answer (the budget
/// applies as for Qwen). Off: an empty think block. Both are written as the
/// template writes past reasoning (`<think>…</think>`, no newlines), so the next
/// turn's prompt reuses this turn's cached prefix. Other templates switch
/// thinking themselves.
pub fn generation_prefix(template: &ChatTemplate, vars: &Map<String, Json>) -> &'static str {
    if !template.opens_thinking() {
        return "";
    }
    match vars.get("enable_thinking") {
        Some(Json::Bool(false)) => "<think></think>",
        _ => "<think>",
    }
}

/// What closes reasoning cut off by the budget: Qwen's template spacing, or
/// the bare `</think>` Flint's and Spark's templates write after past reasoning.
fn reasoning_close(variant: Variant) -> &'static str {
    match variant {
        Variant::Flint | Variant::Spark | Variant::Lfm3 => "</think>",
        Variant::Dense | Variant::Moe => "\n</think>\n\n",
    }
}

/// Map thinking controls onto what this template understands:
/// `reasoning_effort` "none"/"off"/"disabled" becomes `enable_thinking=false`;
/// other efforts snap to the nearest level the template names (e.g. "high"
/// becomes "xhigh" for a template that only knows low/medium/xhigh).
pub fn normalize_thinking(template: &str, vars: &mut Map<String, Json>) {
    for key in ["enable_thinking", "thinking"] {
        if let Some(Json::String(s)) = vars.get(key) {
            let on = !matches!(
                s.to_ascii_lowercase().as_str(),
                "false" | "off" | "no" | "0" | "disabled"
            );
            vars.insert(key.into(), Json::Bool(on));
        }
    }
    let Some(effort) = vars
        .get("reasoning_effort")
        .and_then(Json::as_str)
        .map(str::to_ascii_lowercase)
    else {
        return;
    };
    if matches!(effort.as_str(), "none" | "off" | "disabled" | "false") {
        vars.remove("reasoning_effort");
        vars.insert("enable_thinking".into(), Json::Bool(false));
        return;
    }
    const LEVELS: [&str; 8] = [
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "extreme",
        "ultracode",
    ];
    let known: Vec<usize> = LEVELS
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            template.contains(&format!("'{l}'")) || template.contains(&format!("\"{l}\""))
        })
        .map(|(i, _)| i)
        .collect();
    let Some(wanted) = LEVELS.iter().position(|l| *l == effort) else {
        return;
    };
    if known.is_empty() || known.contains(&wanted) {
        vars.insert("reasoning_effort".into(), Json::String(effort));
        return;
    }
    // Nearest named level, preferring the higher one on ties.
    let nearest = known
        .iter()
        .min_by_key(|&&i| (i as isize - wanted as isize).abs() * 2 - isize::from(i > wanted))
        .copied()
        .unwrap_or(wanted);
    vars.insert(
        "reasoning_effort".into(),
        Json::String(LEVELS[nearest].into()),
    );
}

/// The GGUF's recommended sampling (`general.sampling.*`), with Qwen defaults.
/// The model's recommended sampling: a GGUF's `general.sampling.*`
/// metadata, or an MLX checkpoint's `generation_config.json`.
pub fn recommended_sampling_for(path: &Path) -> Result<SamplingParams> {
    if !super::mlx::is_mlx_dir(path) {
        return Ok(recommended_sampling(&GgufFile::open(path)?));
    }
    let g: serde_json::Value = std::fs::read(path.join("generation_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    Ok(SamplingParams {
        temperature: g["temperature"].as_f64().unwrap_or(1.0) as f32,
        top_p: g["top_p"].as_f64().unwrap_or(0.95) as f32,
        top_k: g["top_k"].as_u64().unwrap_or(20) as usize,
        min_p: g["min_p"].as_f64().unwrap_or(0.) as f32,
        ..SamplingParams::default()
    })
}

pub fn recommended_sampling(file: &GgufFile) -> SamplingParams {
    let md = file.metadata();
    let float = |key: &str| match md.get(key) {
        Some(MetadataValue::Float32(v)) => Some(*v),
        Some(MetadataValue::Float64(v)) => Some(*v as f32),
        _ => None,
    };
    let arch = match md.get("general.architecture") {
        Some(MetadataValue::String(a)) => a.as_str(),
        _ => "",
    };
    SamplingParams {
        temperature: float("general.sampling.temp").unwrap_or(1.0),
        top_p: float("general.sampling.top_p").unwrap_or(0.95),
        top_k: uint(md, "general.sampling.top_k").unwrap_or(20),
        min_p: float("general.sampling.min_p").unwrap_or(0.),
        // Only lfm3 adopts this key. Another file that carries it keeps 1,
        // which is what those models did before the key was read at all.
        repetition_penalty: if arch == "lfm3" {
            float("general.sampling.penalty_repeat").unwrap_or(1.0)
        } else {
            1.0
        },
        ..SamplingParams::default()
    }
}

/// Routes streamed text: reasoning until `</think>`, then content, holding
/// back anything that may start `<tool_call>` (tool calls are reported once
/// complete) or a stop string.
struct Splitter {
    text: String,
    in_reasoning: bool,
    /// Byte offset in `text` up to which deltas were emitted.
    emitted: usize,
    content_start: usize,
    tool_started: bool,
    stop: Vec<String>,
    stopped: bool,
    stop_at: Option<usize>,
    tools: bool,
}

impl Splitter {
    fn new(in_reasoning: bool, stop: &[String], tools: bool) -> Self {
        Self {
            text: String::new(),
            in_reasoning,
            emitted: 0,
            content_start: 0,
            tool_started: false,
            stop: stop.iter().filter(|s| !s.is_empty()).cloned().collect(),
            stopped: false,
            stop_at: None,
            tools,
        }
    }

    fn push(&mut self, piece: &str) -> Vec<Delta> {
        self.text.push_str(piece);
        self.drain(false)
    }

    fn finish(&mut self) -> Vec<Delta> {
        self.drain(true)
    }

    fn visible_content(&self) -> String {
        let end = self.stop_at.unwrap_or(self.text.len());
        self.text[self.content_start.min(end)..end]
            .trim()
            .to_owned()
    }

    fn drain(&mut self, last: bool) -> Vec<Delta> {
        let mut out = Vec::new();
        if self.in_reasoning {
            if let Some(end) = self.text[self.emitted..]
                .find("</think>")
                .map(|i| i + self.emitted)
            {
                if end > self.emitted {
                    out.push(Delta::Reasoning(self.text[self.emitted..end].to_owned()));
                }
                self.in_reasoning = false;
                self.emitted = end + "</think>".len();
                self.content_start = self.emitted;
            } else {
                let safe = if last {
                    self.text.len()
                } else {
                    hold_back(&self.text, self.emitted, &["</think>"])
                };
                if safe > self.emitted {
                    out.push(Delta::Reasoning(self.text[self.emitted..safe].to_owned()));
                    self.emitted = safe;
                }
                return out;
            }
        }
        if self.tool_started || self.stopped {
            return out;
        }
        let pending = &self.text[self.emitted..];
        let stop_hit = self
            .stop
            .iter()
            .filter_map(|s| pending.find(s.as_str()))
            .min();
        let tool_hit = if self.tools {
            pending.find("<tool_call>")
        } else {
            None
        };
        let mut end = if last {
            self.text.len()
        } else {
            let mut markers: Vec<&str> = if self.tools {
                vec!["<tool_call>"]
            } else {
                Vec::new()
            };
            markers.extend(self.stop.iter().map(String::as_str));
            hold_back(&self.text, self.emitted, &markers)
        };
        if let Some(t) = tool_hit {
            end = end.max(self.emitted).min(self.emitted + t);
            self.tool_started = true;
        }
        if let Some(s) = stop_hit
            && tool_hit.is_none_or(|t| s < t)
        {
            end = self.emitted + s;
            self.stopped = true;
            self.stop_at = Some(end);
            self.tool_started = false;
        }
        // Leading whitespace right after </think> is template formatting.
        let mut begin = self.emitted;
        if begin == self.content_start {
            let trimmed = self.text[begin..end.max(begin)].trim_start();
            if trimmed.is_empty() && !last && !self.stopped && !self.tool_started {
                return out;
            }
            begin = end.max(begin) - trimmed.len();
        }
        if end > begin {
            out.push(Delta::Content(self.text[begin..end].to_owned()));
        }
        self.emitted = end.max(self.emitted);
        out
    }
}

/// Largest offset >= `from` such that `text[..offset]` cannot be the start of
/// any marker still being generated.
fn hold_back(text: &str, from: usize, markers: &[&str]) -> usize {
    let mut safe = text.len();
    for marker in markers {
        for k in (1..marker.len()).rev() {
            if text.len() >= k
                && text.is_char_boundary(text.len() - k)
                && marker.starts_with(&text[text.len() - k..])
            {
                safe = safe.min(text.len() - k);
                break;
            }
        }
    }
    while safe > from && !text.is_char_boundary(safe) {
        safe -= 1;
    }
    safe.max(from)
}

const MEDIA_KINDS: [&str; 2] = ["image", "audio"];

fn media_mismatch(kind: usize) -> Error {
    let what = MEDIA_KINDS[kind];
    Error::Parameter(format!(
        "the chat template's {what} placeholders do not match the attached {what}"
    ))
}

/// Repeat each placeholder in `prompt` once per token of its item. Kind 0 is
/// images, kind 1 audio clips; each kind keeps its own order and the prompt
/// interleaves them. An empty marker is never matched.
fn splice_media(prompt: &str, markers: [&str; 2], lengths: [&[usize]; 2]) -> Result<String> {
    let mut next = [0usize; 2];
    let mut text = String::with_capacity(prompt.len());
    let mut rest = prompt;
    loop {
        let found = markers
            .iter()
            .enumerate()
            .filter(|(_, marker)| !marker.is_empty())
            .filter_map(|(kind, marker)| rest.find(marker).map(|at| (at, kind)))
            .min();
        let Some((at, kind)) = found else { break };
        let len = *lengths[kind]
            .get(next[kind])
            .ok_or_else(|| media_mismatch(kind))?;
        next[kind] += 1;
        text.push_str(&rest[..at]);
        text.push_str(&markers[kind].repeat(len));
        rest = &rest[at + markers[kind].len()..];
    }
    text.push_str(rest);
    for kind in 0..2 {
        if next[kind] != lengths[kind].len() {
            return Err(media_mismatch(kind));
        }
    }
    Ok(text)
}

/// Find each item's run of pad tokens in `tokens`: `(start, kind, index)` in
/// prompt order. Clips have no wrapper tokens, so two in a row make one run
/// of pads, and each takes exactly its own length from it; an image's run
/// must end where the image does.
fn locate_media(
    tokens: &[u32],
    pads: [Option<u32>; 2],
    lengths: [&[usize]; 2],
) -> Result<Vec<(usize, usize, usize)>> {
    let mut runs = Vec::with_capacity(lengths[0].len() + lengths[1].len());
    let mut taken = [0usize; 2];
    let mut i = 0;
    while i < tokens.len() {
        let Some(kind) = (0..2).find(|&k| pads[k] == Some(tokens[i])) else {
            i += 1;
            continue;
        };
        let pad = tokens[i];
        let start = i;
        let len = *lengths[kind]
            .get(taken[kind])
            .ok_or_else(|| media_mismatch(kind))?;
        runs.push((start, kind, taken[kind]));
        taken[kind] += 1;
        i += len;
        if i > tokens.len() || tokens[start..i].iter().any(|&t| t != pad) {
            return Err(media_mismatch(kind));
        }
        if kind == 0 && tokens.get(i) == Some(&pad) {
            return Err(media_mismatch(kind));
        }
    }
    for kind in 0..2 {
        if taken[kind] != lengths[kind].len() {
            return Err(media_mismatch(kind));
        }
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splices_and_locates_adjacent_clips_and_images() {
        let text = splice_media("a<A><I>b<A>c", ["<I>", "<A>"], [&[2], &[3, 1]]).unwrap();
        assert_eq!(text, "a<A><A><A><I><I>b<A>c");
        // Tokens: text 0, image pad 7, audio pad 9. Two clips back to back
        // form one run of pads and still split by their own lengths.
        let tokens = [0, 9, 9, 9, 9, 7, 7, 0];
        let runs = locate_media(&tokens, [Some(7), Some(9)], [&[2], &[3, 1]]).unwrap();
        assert_eq!(runs, vec![(1, 1, 0), (4, 1, 1), (5, 0, 0)]);
    }

    #[test]
    fn media_counts_and_runs_must_match() {
        // More placeholders than clips, and fewer.
        assert!(splice_media("<A><A>", ["<I>", "<A>"], [&[], &[1]]).is_err());
        assert!(splice_media("<A>", ["<I>", "<A>"], [&[], &[1, 1]]).is_err());
        // No audio marker (no projector): the text is left alone, but a clip
        // with nowhere to go is an error.
        assert_eq!(splice_media("<A>", ["<I>", ""], [&[], &[]]).unwrap(), "<A>");
        assert!(splice_media("x", ["<I>", ""], [&[], &[2]]).is_err());
        // An image run longer than its image.
        assert!(locate_media(&[7, 7, 7], [Some(7), None], [&[2], &[]]).is_err());
        // A clip run shorter than its clip.
        assert!(locate_media(&[9, 9, 0], [None, Some(9)], [&[], &[3]]).is_err());
        // A clip left over at the end.
        assert!(locate_media(&[9, 0], [None, Some(9)], [&[], &[1, 1]]).is_err());
    }

    fn run(pieces: &[&str], reasoning: bool, stop: &[&str]) -> (Vec<Delta>, Splitter) {
        let mut s = Splitter::new(
            reasoning,
            &stop.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            true,
        );
        let mut out = Vec::new();
        for p in pieces {
            out.extend(s.push(p));
        }
        out.extend(s.finish());
        (out, s)
    }

    fn join(deltas: &[Delta]) -> (String, String) {
        let mut r = String::new();
        let mut c = String::new();
        for d in deltas {
            match d {
                Delta::Reasoning(t) => r.push_str(t),
                Delta::Content(t) => c.push_str(t),
            }
        }
        (r, c)
    }

    #[test]
    fn splits_reasoning_content_and_holds_back_tool_calls() {
        let (deltas, _) = run(
            &[
                "Think",
                "ing.</th",
                "ink>\n\nHel",
                "lo <tool_",
                "call>\n<function=f>",
            ],
            true,
            &[],
        );
        let (r, c) = join(&deltas);
        assert_eq!(r, "Thinking.");
        assert_eq!(c, "Hello ");
    }

    /// The owned stream decodes as `tokenizers`' `DecodeStream`, holding back
    /// a character split across tokens.
    #[test]
    fn text_stream_matches_decode_stream() {
        let json = serde_json::json!({
            "version": "1.0",
            "model": {"type": "WordLevel", "unk_token": "a",
                      "vocab": {"a": 0, "\u{c3}": 1, "\u{a9}": 2, "\u{120}b": 3}},
            "decoder": {"type": "ByteLevel", "add_prefix_space": false,
                        "trim_offsets": false, "use_regex": false},
        });
        let tokenizer = Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap();
        let ids = [0, 1, 2, 3, 0, 1, 2];
        let mut reference = tokenizer.decode_stream();
        let mut ours = TextStream::default();
        for id in ids {
            assert_eq!(
                ours.step(&tokenizer, id).unwrap(),
                reference.step(id).unwrap()
            );
        }
        assert_eq!(tokenizer.decode(&ids).unwrap(), "a\u{e9} ba\u{e9}");
    }

    #[test]
    fn stop_strings_truncate_content() {
        let (deltas, s) = run(&["a b ST", "OP c"], false, &["STOP"]);
        assert_eq!(join(&deltas).1, "a b ");
        assert!(s.stopped);
        assert_eq!(s.visible_content(), "a b");
    }
}

#[cfg(test)]
mod thinking_tests {
    use super::*;

    /// Flint's template has no thinking switch, so the prompt opens or closes
    /// the think block; Qwen's template switches itself and needs nothing,
    /// even when its source never says "thinking".
    #[test]
    fn templates_without_a_switch_open_or_close_thinking() {
        let flint = ChatTemplate::new(include_str!(
            "../../tests/fixtures/flint/chat_template.jinja"
        ))
        .unwrap()
        .with_opens_thinking(true);
        let qwen = ChatTemplate::new(
            "{%- if enable_thinking is not defined or enable_thinking %}think{% endif %}",
        )
        .unwrap();
        let mut off = Map::new();
        off.insert("enable_thinking".into(), Json::Bool(false));
        let mut on = Map::new();
        on.insert("enable_thinking".into(), Json::Bool(true));
        assert_eq!(generation_prefix(&flint, &off), "<think></think>");
        assert_eq!(generation_prefix(&flint, &on), "<think>");
        assert_eq!(generation_prefix(&flint, &Map::new()), "<think>");
        assert_eq!(generation_prefix(&qwen, &off), "");
        assert_eq!(generation_prefix(&qwen, &Map::new()), "");
        let plain = ChatTemplate::new("{{ messages[0].content }}").unwrap();
        assert_eq!(generation_prefix(&plain, &Map::new()), "");
        assert_eq!(reasoning_close(Variant::Flint), "</think>");
        assert_eq!(reasoning_close(Variant::Spark), "</think>");
        assert_eq!(reasoning_close(Variant::Lfm3), "</think>");
        assert_eq!(reasoning_close(Variant::Dense), "\n</think>\n\n");
    }

    #[test]
    fn effort_snaps_to_template_levels() {
        let template = "{% if reasoning_effort not in ('xhigh', 'medium', 'low') %}";
        let mut vars = Map::new();
        vars.insert("reasoning_effort".into(), Json::String("high".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["reasoning_effort"], "xhigh");
        vars.insert("reasoning_effort".into(), Json::String("minimal".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["reasoning_effort"], "low");
        vars.insert("reasoning_effort".into(), Json::String("none".into()));
        normalize_thinking(template, &mut vars);
        assert_eq!(vars["enable_thinking"], false);
        assert!(!vars.contains_key("reasoning_effort"));
    }
}
