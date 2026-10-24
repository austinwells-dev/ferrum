// Copyright 2026 Nace AI (Kapitsa, https://github.com/nace-ai/kapitsa)
// Copyright 2026 Austin Wells (Ferrum changes)
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// This file is Apache-2.0 only (it is not part of Ferrum's MIT dual license).
// Adapted from Kapitsa's KFlash budget (src/model/kflash_budget.{h,cpp},
// KfBudget and KfCosts, at be57c8f) and its tree builder's calibration and
// row rule (src/kernels/kflash.h: kf_lcal, kf_build_impl's E(n) and choice
// of n). Changes: Rust on the host instead of the GPU-side builder; a cycle's
// cost is its wall time (drafting and verify) rather than GPU time; no wide /
// narrow capacity decision (Ferrum's verify picks its kernels from the row
// count alone); the allowed row counts are a fixed ladder; the cache lives in
// ~/.cache/ferrum/kflash.

//! KFlash's tree budget: how many rows a draft tree's verify checks each
//! cycle, learned online from the cycles themselves.
//!
//! - Calibration: the drafter's probability q of a node is mapped to the
//!   probability that the target accepts it, p = sigmoid(alpha_d + beta_d
//!   logit(q)) per depth d, fitted by a regularized logistic regression on
//!   the latest records: every verified node whose parent was accepted, and
//!   whether it was. Before any records p = q.
//! - Cost: a cycle's milliseconds by the rows it verifies, a base that every
//!   cycle moves (the context's length) plus each count's own offset. The
//!   first cycles try every allowed count a few times.
//! - Rule: the drafter builds its tree best first by calibrated path
//!   probability, so its first n - 1 nodes are its best tree of n rows, which
//!   is expected to emit E(n) = 1 + the summed path probabilities of those
//!   nodes. The cycle verifies the allowed n that maximizes E(n) - rate
//!   cost(n), rate being the recent cycles' tokens per millisecond (a row
//!   must pay at the long-run rate); before 16 cycles, E(n) / cost(n).
//!
//! The table and the calibration persist per model, drafter and chip.
#![forbid(unsafe_code)]
use std::path::PathBuf;

/// Most verify rows a tree may take (the anchor and 31 nodes).
pub const MAX_ROWS: usize = 32;
const D: usize = MAX_ROWS + 1;
/// Records kept per depth.
const WINDOW: usize = 2048;
/// Cycles per allowed count before the rule decides.
const EXPLORE: usize = 3;
/// Cycles between saves after a request (a process killed hard loses less).
pub const SAVE_EVERY: usize = 256;
/// The row counts a cycle may verify: Ferrum's verify GEMMs run whole tiles,
/// so a count between these costs about as much as the next one up.
const LADDER: [usize; 7] = [2, 4, 8, 12, 16, 24, 32];
/// What the cache's costs measure (a table measured as anything else is
/// measured again; the calibration is kept).
const COST_OF: &str = "cycle wall ms";

/// Per-depth calibration of a drafter's node probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    alpha: [f32; D],
    beta: [f32; D],
}

impl Default for Calibration {
    /// The identity: p = q.
    fn default() -> Self {
        Self {
            alpha: [0.; D],
            beta: [1.; D],
        }
    }
}

impl Calibration {
    /// Log of the calibrated acceptance probability of a node at `depth`
    /// (1 = a child of the anchor) whose drafter log-probability is `lq`.
    pub fn log_p(&self, depth: usize, lq: f64) -> f64 {
        let d = depth.clamp(1, MAX_ROWS);
        let q = lq.exp().min(0.999_999);
        let z = f64::from(self.alpha[d]) + f64::from(self.beta[d]) * (lq - (1. - q).ln());
        -softplus(-z)
    }
}

fn softplus(x: f64) -> f64 {
    if x > 20. { x } else { x.exp().ln_1p() }
}

fn logit_q(lq: f32) -> f32 {
    let q = f64::from(lq).exp().min(0.999_999);
    (f64::from(lq) - (-q).ln_1p()) as f32
}

/// L2-regularized logistic regression of y on (1, x) toward the identity
/// (alpha 0, beta 1): Newton steps from the current values. The prior weighs
/// about as much as four records; beta stays positive (the ranking of
/// siblings is the drafter's).
fn fit(x: &[f32], y: &[u8], alpha: &mut f32, beta: &mut f32, iters: usize) {
    const LAMBDA: f64 = 1.;
    let (mut a, mut b) = (f64::from(*alpha), f64::from(*beta));
    for _ in 0..iters {
        let (mut ga, mut gb) = (LAMBDA * a, LAMBDA * (b - 1.));
        let (mut haa, mut hab, mut hbb) = (LAMBDA, 0., LAMBDA);
        for (&xi, &yi) in x.iter().zip(y) {
            let xi = f64::from(xi);
            let p = 1. / (1. + (-(a + b * xi)).exp());
            let (w, r) = (p * (1. - p), p - f64::from(yi));
            ga += r;
            gb += r * xi;
            haa += w;
            hab += w * xi;
            hbb += w * xi * xi;
        }
        let det = haa * hbb - hab * hab;
        if det.is_nan() || det <= 1e-12 {
            break;
        }
        let da = (hbb * ga - hab * gb) / det;
        let db = (haa * gb - hab * ga) / det;
        a = (a - da).clamp(-8., 8.);
        b = (b - db).clamp(0.1, 4.);
        if da.abs() + db.abs() < 1e-4 {
            break;
        }
    }
    *alpha = a as f32;
    *beta = b as f32;
}

/// A cycle's cost by the rows it verifies: a base that every cycle moves plus
/// each count's offset above it, a running mean of what its cycles cost above
/// the base. A process sets the base by its first cycle; with a table loaded
/// from the cache, by its first cycle of a count the cache holds (until then
/// the counts it measures keep absolute offsets, `pre`).
#[derive(Debug, Clone)]
struct Costs {
    base: f64,
    off: [f64; D],
    based: bool,
    loaded: bool,
    pre: u64,
    count: [usize; D],
}

impl Default for Costs {
    fn default() -> Self {
        Self {
            base: 0.,
            off: [0.; D],
            based: false,
            loaded: false,
            pre: 0,
            count: [0; D],
        }
    }
}

impl Costs {
    fn observe(&mut self, n: usize, ms: f64) {
        let pre = |m: usize| (self.pre >> m) & 1 == 1;
        if !self.based && (!self.loaded || (self.count[n] > 0 && !pre(n))) {
            self.base = if self.count[n] > 0 {
                ms - self.off[n]
            } else {
                ms
            };
            if self.count[n] == 0 {
                self.off[n] = 0.;
                self.count[n] = 1;
            }
            for m in 1..D {
                if (self.pre >> m) & 1 == 1 {
                    self.off[m] -= self.base;
                }
            }
            self.pre = 0;
            self.based = true;
        } else if !self.based {
            // A count the cache does not hold, before the base: absolute for now.
            self.off[n] = if self.count[n] == 0 {
                ms
            } else {
                self.off[n] + (ms - self.off[n]) / (self.count[n] + 1).min(32) as f64
            };
            self.count[n] += 1;
            self.pre |= 1 << n;
        } else if self.count[n] == 0 {
            self.off[n] = ms - self.base;
            self.count[n] = 1;
        } else {
            let e = ms - (self.base + self.off[n]);
            self.base += 0.5 * e;
            self.off[n] += e / (self.count[n] + 1).min(32) as f64;
            self.count[n] += 1;
        }
    }

    /// n rows' cost now (0: not measured).
    fn at(&self, n: usize) -> f64 {
        if self.count[n] > 0 {
            self.base + self.off[n]
        } else {
            0.
        }
    }

    /// The count the next cycle must verify, if any: with a loaded table,
    /// before this process's base, the largest count the table holds; then
    /// every allowed count `EXPLORE` times.
    fn force(&self, allowed: &[usize]) -> Option<usize> {
        if !self.based
            && self.loaded
            && allowed.len() > 1
            && let Some(&n) = allowed
                .iter()
                .rev()
                .find(|&&n| self.count[n] > 0 && (self.pre >> n) & 1 == 0)
        {
            return Some(n);
        }
        allowed.iter().copied().find(|&n| self.count[n] < EXPLORE)
    }

    fn table(&self) -> Vec<f64> {
        (0..D)
            .map(|n| {
                if n > 0 && self.count[n] > 0 && (self.pre >> n) & 1 == 0 {
                    (self.base + self.off[n]).max(0.01)
                } else {
                    0.
                }
            })
            .collect()
    }

    fn load(&mut self, cost: &[f64]) {
        for (n, &c) in cost.iter().enumerate().take(D).skip(1) {
            if c > 0. {
                self.off[n] = c;
                self.count[n] = EXPLORE;
                self.loaded = true;
            }
        }
    }
}

/// What the rule decides a cycle with, from past cycles only.
#[derive(Debug, Clone)]
pub struct CyclePlan {
    pub calibration: Calibration,
    /// Verify exactly this many rows (exploration).
    force: Option<usize>,
    /// Tokens per millisecond of the recent cycles (None: too few yet).
    rate: Option<f64>,
    /// Allowed row counts and their costs.
    allowed: Vec<(usize, f64)>,
}

impl CyclePlan {
    /// Verify rows for a tree whose nodes, in build order, have calibrated
    /// path log-probabilities `scores` (non-increasing): the allowed count n
    /// that maximizes E(n) - rate cost(n), and E(n). A tree without scores
    /// verifies all its nodes.
    pub fn rows(&self, scores: &[f32]) -> (usize, f64) {
        let built = scores.len() + 1;
        let expected = |n: usize| {
            1. + scores[..n - 1]
                .iter()
                .map(|&s| f64::from(s).exp())
                .sum::<f64>()
        };
        if let Some(n) = self.force {
            let n = n.min(built);
            return (n, expected(n));
        }
        let mut best = (built, f64::NEG_INFINITY);
        for &(n, cost) in &self.allowed {
            if n > built {
                break;
            }
            let e = expected(n);
            let value = match self.rate {
                Some(rate) => e - rate * cost,
                None => e / cost,
            };
            if value > best.1 {
                best = (n, value);
            }
        }
        (best.0, expected(best.0))
    }
}

/// One verified node whose parent was on the accepted path.
#[derive(Debug, Clone, Copy)]
pub struct Record {
    pub depth: usize,
    /// The drafter's log-probability of the node given its parent.
    pub lq: f32,
    pub accepted: bool,
}

/// The budget across cycles (see the module docs).
pub struct Budget {
    path: Option<PathBuf>,
    allowed: Vec<usize>,
    x: Vec<Vec<f32>>,
    y: Vec<Vec<u8>>,
    head: [usize; D],
    calibration: Calibration,
    costs: Costs,
    rate_tokens: f64,
    rate_ms: f64,
    rate_cycles: usize,
    cycles: usize,
    since_fit: usize,
    saved: usize,
    by_rows: [usize; D],
    rows_sum: usize,
    expected_sum: f64,
    tokens_sum: f64,
}

impl Budget {
    /// A budget for trees of at most `max_rows` verify rows. `key` names its
    /// cache file (model, drafter, chip); None keeps nothing.
    pub fn new(key: Option<&str>, max_rows: usize) -> Self {
        let max_rows = max_rows.clamp(2, MAX_ROWS);
        let mut allowed: Vec<usize> = LADDER.iter().copied().filter(|&n| n < max_rows).collect();
        allowed.push(max_rows);
        let path = key.and_then(|key| {
            let key: String = key
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            cache_dir().map(|dir| dir.join(format!("{key}.json")))
        });
        let mut budget = Self {
            path,
            allowed,
            x: vec![Vec::new(); D],
            y: vec![Vec::new(); D],
            head: [0; D],
            calibration: Calibration::default(),
            costs: Costs::default(),
            rate_tokens: 0.,
            rate_ms: 0.,
            rate_cycles: 0,
            cycles: 0,
            since_fit: 0,
            saved: 0,
            by_rows: [0; D],
            rows_sum: 0,
            expected_sum: 0.,
            tokens_sum: 0.,
        };
        budget.load();
        budget
    }

    /// The next cycle's calibration and rule.
    pub fn plan(&self) -> CyclePlan {
        CyclePlan {
            calibration: self.calibration.clone(),
            force: self.costs.force(&self.allowed),
            rate: (self.rate_cycles >= 16 && self.rate_ms > 0.)
                .then(|| self.rate_tokens / self.rate_ms),
            allowed: self
                .allowed
                .iter()
                .map(|&n| {
                    let c = self.costs.at(n);
                    (n, if c > 0. { c.max(0.01) } else { 1. })
                })
                .collect(),
        }
    }

    /// After a cycle that verified `rows` rows (expected to emit `expected`
    /// tokens), accepted `accepted` drafts and took `ms`.
    pub fn observe(
        &mut self,
        rows: usize,
        expected: f64,
        accepted: usize,
        ms: f64,
        records: &[Record],
    ) {
        let n = rows.clamp(1, MAX_ROWS);
        self.cycles += 1;
        self.by_rows[n] += 1;
        self.rows_sum += n;
        self.expected_sum += expected;
        self.tokens_sum += (accepted + 1) as f64;
        self.costs.observe(n, ms);
        self.rate_tokens = self.rate_tokens * (63. / 64.) + (accepted + 1) as f64;
        self.rate_ms = self.rate_ms * (63. / 64.) + ms;
        self.rate_cycles += 1;
        for r in records {
            if (1..=MAX_ROWS).contains(&r.depth) {
                self.add_record(r.depth, r.lq, r.accepted);
            }
        }
        self.since_fit += 1;
        if self.since_fit >= 16 {
            self.since_fit = 0;
            for d in 1..D {
                if self.x[d].len() >= 32 {
                    fit(
                        &self.x[d],
                        &self.y[d],
                        &mut self.calibration.alpha[d],
                        &mut self.calibration.beta[d],
                        4,
                    );
                }
            }
        }
    }

    fn add_record(&mut self, d: usize, lq: f32, hit: bool) {
        let (x, y) = (logit_q(lq), u8::from(hit));
        if self.x[d].len() < WINDOW {
            self.x[d].push(x);
            self.y[d].push(y);
        } else {
            self.x[d][self.head[d]] = x;
            self.y[d][self.head[d]] = y;
            self.head[d] = (self.head[d] + 1) % WINDOW;
        }
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    /// For logs: rows per cycle, the cost table and the calibration.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "KFlash: {} cycles, {:.2} rows per cycle; cycles by rows",
            self.cycles,
            self.rows_sum as f64 / self.cycles.max(1) as f64
        );
        for n in 1..D {
            if self.by_rows[n] > 0 {
                s += &format!(" {n}:{}", self.by_rows[n]);
            }
        }
        if self.cycles > 0 {
            s += &format!(
                "; tokens per cycle expected {:.3}, got {:.3}",
                self.expected_sum / self.cycles as f64,
                self.tokens_sum / self.cycles as f64
            );
        }
        if self.rate_ms > 0. {
            s += &format!("; rate {:.4} tok/ms", self.rate_tokens / self.rate_ms);
        }
        s += "; cost ms by rows";
        for &n in &self.allowed {
            s += &format!(" {n}:{:.1}", self.costs.at(n));
        }
        s += "; calibration (alpha, beta) by depth";
        for d in 1..=8 {
            s += &format!(
                " ({:.2}, {:.2})",
                self.calibration.alpha[d], self.calibration.beta[d]
            );
        }
        s
    }

    fn load(&mut self) {
        let Some(path) = &self.path else { return };
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let Ok(j) = serde_json::from_str::<serde_json::Value>(&text) else {
            eprintln!("KFlash: ignoring an unreadable cache {}", path.display());
            return;
        };
        let floats = |key: &str| -> Vec<f64> {
            j.get(key)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(0.)).collect())
                .unwrap_or_default()
        };
        if j.get("cost_of").and_then(|v| v.as_str()) == Some(COST_OF) {
            self.costs.load(&floats("cost"));
        }
        let (alpha, beta) = (floats("alpha"), floats("beta"));
        for d in 1..D.min(alpha.len()).min(beta.len()) {
            self.calibration.alpha[d] = alpha[d] as f32;
            self.calibration.beta[d] = beta[d] as f32;
        }
    }

    /// Write the table and the calibration to the cache, once 32 cycles were
    /// observed here and `fresh` of them since the last save.
    pub fn save(&mut self, fresh: usize) {
        let Some(path) = &self.path else { return };
        if self.cycles < 32 || self.cycles - self.saved < fresh.max(1) {
            return;
        }
        let j = serde_json::json!({
            "cost_of": COST_OF,
            "cost": self.costs.table(),
            "alpha": self.calibration.alpha.to_vec(),
            "beta": self.calibration.beta.to_vec(),
        });
        // Written whole beside its place, then renamed over it.
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&tmp, format!("{j}\n")))
            .and_then(|()| std::fs::rename(&tmp, path));
        match written {
            Ok(()) => self.saved = self.cycles,
            Err(e) => {
                let _ = std::fs::remove_file(&tmp);
                eprintln!("KFlash: could not write {}: {e}", path.display());
            }
        }
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        self.save(1);
    }
}

/// `$FERRUM_HOME/kflash`, else `~/.cache/ferrum/kflash`.
fn cache_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("FERRUM_HOME") {
        return Some(PathBuf::from(home).join("kflash"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/ferrum/kflash"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_identity_calibration_keeps_q() {
        let c = Calibration::default();
        for lq in [-5f64, -1., -0.1, -0.001] {
            assert!((c.log_p(3, lq) - lq).abs() < 1e-6, "{lq}");
        }
    }

    #[test]
    fn the_fit_recovers_an_overconfident_drafter() {
        // Accepted with probability q / 2: alpha < 0 at fixed beta ~ 1.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut state = 1u64;
        for i in 0..2000 {
            let q = 0.05 + 0.9 * (i % 100) as f32 / 100.;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 40) as f32 / (1u64 << 24) as f32;
            x.push(logit_q(q.ln()));
            y.push(u8::from(u < q / 2.));
        }
        let (mut a, mut b) = (0., 1.);
        fit(&x, &y, &mut a, &mut b, 20);
        assert!(a < -0.4, "alpha {a}, beta {b}");
        let c = Calibration {
            alpha: [a; D],
            beta: [b; D],
        };
        let p = c.log_p(1, 0.5f64.ln()).exp();
        assert!((p - 0.25).abs() < 0.08, "p(0.5) = {p}");
    }

    #[test]
    fn exploration_then_the_rule() {
        let mut b = Budget::new(None, 16);
        assert_eq!(b.allowed, vec![2, 4, 8, 12, 16]);
        // Every count is verified EXPLORE times, smallest first.
        let mut seen = Vec::new();
        while let Some(n) = b.plan().force {
            seen.push(n);
            let ms = if n <= 8 { 150. } else { 170. };
            b.observe(n, 1., 1, ms, &[]);
        }
        assert_eq!(seen.len(), 5 * EXPLORE);
        // Sure nodes: 16 rows pay; doubtful ones past the 8th do not.
        let sure = vec![-0.01f32; 15];
        assert_eq!(b.plan().rows(&sure).0, 16);
        let mut doubtful = vec![-0.01f32; 7];
        doubtful.extend([-6f32; 8]);
        assert_eq!(b.plan().rows(&doubtful).0, 8);
    }
}
