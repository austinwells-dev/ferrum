//! Text-to-image generation with Qwen-Image-2.1 and its Turbo distillation.
//!
//! Three models run one after the other (`pipeline`): a Qwen3-VL text encoder
//! turns the prompt into a sequence of embeddings, a single-stream diffusion
//! transformer (`dit`) denoises a grid of 64-channel latents for a handful of
//! flow-matching Euler steps, and a convolutional decoder (`vae`) turns the
//! latents into pixels. Only one of them is resident at a time, which is what
//! lets a 7B transformer and an 8B encoder share a 32 GB Mac.
//!
//! The text tokens are modulated from `t = 0` and attend causally, so their
//! keys and values are the same at every step. The transformer computes them
//! once, and each denoising step then runs only the image tokens.
#![forbid(unsafe_code)]
pub mod checkpoint;
pub mod config;
pub mod dit;
pub(crate) mod ops;
#[cfg(test)]
mod reference_tests;
pub mod rope;
pub mod schedule;
#[cfg(test)]
mod tests;

pub use config::{DitConfig, ModelDir, VaeConfig};
