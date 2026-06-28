//! Burn-based training for KGE models.
//!
//! Alternative to the candle-based [`crate::train`] module. Supports
//! CPU (ndarray + rayon) and GPU (WGPU/Metal/Vulkan) backends via
//! burn's backend system.
//!
//! Enable with `burn-ndarray` (ndarray) or `burn-wgpu` (WGPU) feature.
//!
//! Two entry points: [`train_complex`] (ComplEx-specific, kept for the bench
//! examples) and [`train_kge`], generic over all four models (TransE, RotatE,
//! ComplEx, DistMult) via [`BurnModelType`] with 1-N (1vsAll) scoring.

use burn::module::{Module, Param, ParamId};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation;
use burn::tensor::backend::AutodiffBackend;

/// ComplEx model as a burn Module.
///
/// Stores entity and relation embeddings as separate real/imaginary
/// parameter tensors.
#[derive(Module, Debug)]
pub struct BurnComplEx<B: Backend> {
    entity_re: Param<Tensor<B, 2>>,
    entity_im: Param<Tensor<B, 2>>,
    relation_re: Param<Tensor<B, 2>>,
    relation_im: Param<Tensor<B, 2>>,
}

/// Configuration for burn-based training.
#[derive(Debug, Clone)]
pub struct BurnTrainConfig {
    /// Complex dimension (each embedding stores dim real + dim imaginary).
    pub dim: usize,
    /// Initialization scale (std of normal distribution).
    pub init_scale: f64,
    /// Learning rate.
    pub lr: f64,
    /// Label smoothing epsilon. 0 = no smoothing.
    pub label_smoothing: f64,
    /// N3 regularization coefficient. 0 = disabled.
    pub n3_reg: f64,
    /// Batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Print loss every N epochs. 0 = silent.
    pub log_interval: usize,
}

impl Default for BurnTrainConfig {
    fn default() -> Self {
        Self {
            dim: 200,
            init_scale: 1e-3,
            lr: 0.001,
            label_smoothing: 0.1,
            n3_reg: 0.0,
            batch_size: 512,
            epochs: 100,
            log_interval: 10,
        }
    }
}

/// Result of burn-based training.
pub struct BurnTrainResult {
    /// Entity embeddings as `Vec<Vec<f32>>` (interleaved re/im).
    pub entity_vecs: Vec<Vec<f32>>,
    /// Relation embeddings as `Vec<Vec<f32>>` (interleaved re/im).
    pub relation_vecs: Vec<Vec<f32>>,
    /// Complex dimension.
    pub dim: usize,
    /// Loss per epoch.
    pub losses: Vec<f32>,
}

impl BurnTrainResult {
    /// Convert to a CPU ComplEx scorer for evaluation.
    pub fn to_complex(&self) -> crate::ComplEx {
        crate::ComplEx::from_vecs(
            self.entity_vecs.clone(),
            self.relation_vecs.clone(),
            self.dim,
        )
    }
}

/// Initialize a BurnComplEx model.
fn init_model<B: AutodiffBackend>(
    num_entities: usize,
    num_relations: usize,
    dim: usize,
    init_scale: f64,
    device: &B::Device,
) -> BurnComplEx<B> {
    let mk = |rows, cols| {
        Param::initialized(
            ParamId::new(),
            Tensor::<B, 2>::random(
                [rows, cols],
                burn::tensor::Distribution::Normal(0.0, init_scale),
                device,
            )
            .require_grad(),
        )
    };
    BurnComplEx {
        entity_re: mk(num_entities, dim),
        entity_im: mk(num_entities, dim),
        relation_re: mk(num_relations, dim),
        relation_im: mk(num_relations, dim),
    }
}

/// Score all entities as tails for a batch of (h, r) queries.
///
/// Returns `[batch, num_entities]` where higher = more likely.
fn score_1n<B: Backend>(
    model: &BurnComplEx<B>,
    heads: &Tensor<B, 1, Int>,
    rels: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let h_re = model.entity_re.val().select(0, heads.clone());
    let h_im = model.entity_im.val().select(0, heads.clone());
    let r_re = model.relation_re.val().select(0, rels.clone());
    let r_im = model.relation_im.val().select(0, rels.clone());

    // h * r (complex multiply)
    let hr_re = h_re.clone() * r_re.clone() - h_im.clone() * r_im.clone();
    let hr_im = h_re * r_im + h_im * r_re;

    // Score against all entities: Re(hr * conj(e)) = hr_re @ e_re^T + hr_im @ e_im^T
    let e_re = model.entity_re.val();
    let e_im = model.entity_im.val();
    hr_re.matmul(e_re.transpose()) + hr_im.matmul(e_im.transpose())
}

/// Score all entities as heads for a batch of (r, t) queries.
fn score_1n_heads<B: Backend>(
    model: &BurnComplEx<B>,
    rels: &Tensor<B, 1, Int>,
    tails: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let r_re = model.relation_re.val().select(0, rels.clone());
    let r_im = model.relation_im.val().select(0, rels.clone());
    let t_re = model.entity_re.val().select(0, tails.clone());
    let t_im = model.entity_im.val().select(0, tails.clone());

    // r * conj(t)
    let rc_re = r_re.clone() * t_re.clone() + r_im.clone() * t_im.clone();
    let rc_im = r_im * t_re - r_re * t_im;

    // Re(h * rc) = h_re @ rc_re^T + h_im @ rc_im^T ... but rc is [batch, dim]
    // We need [batch, num_entities], so: rc_re @ e_re^T + rc_im @ e_im^T
    let e_re = model.entity_re.val();
    let e_im = model.entity_im.val();
    rc_re.matmul(e_re.transpose()) + rc_im.matmul(e_im.transpose())
}

/// Train ComplEx with 1-N scoring using burn.
///
/// Returns entity/relation embeddings and per-epoch losses.
pub fn train_complex<B: AutodiffBackend>(
    train_triples: &[crate::dataset::TripleIds],
    num_entities: usize,
    num_relations: usize,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> BurnTrainResult {
    let mut model = init_model::<B>(
        num_entities,
        num_relations,
        config.dim,
        config.init_scale,
        device,
    );
    let mut optim = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .init::<B, BurnComplEx<B>>();

    let n_triples = train_triples.len();
    let batch_size = config.batch_size.min(n_triples);
    let eps = config.label_smoothing;
    let mut losses = Vec::with_capacity(config.epochs);

    let mut indices: Vec<usize> = (0..n_triples).collect();

    for epoch in 0..config.epochs {
        let epoch_start = std::time::Instant::now();

        // Shuffle.
        {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::rng());
        }

        let mut epoch_loss = 0.0_f64;
        let mut n_batches = 0u32;
        let mut offset = 0;

        while offset < n_triples {
            let end = (offset + batch_size).min(n_triples);
            let batch_idx = &indices[offset..end];
            let actual_bs = batch_idx.len();
            offset = end;

            let heads_data: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].head as i64)
                .collect();
            let rels_data: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].relation as i64)
                .collect();
            let tails_data: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].tail as i64)
                .collect();

            let heads = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(heads_data, [actual_bs]),
                device,
            );
            let rels = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(rels_data, [actual_bs]),
                device,
            );
            let tails = Tensor::<B, 1, Int>::from_data(
                burn::tensor::TensorData::new(tails_data.clone(), [actual_bs]),
                device,
            );

            let current = model.clone();

            // Tail prediction: score all entities for (h, r, ?).
            let tail_scores = score_1n(&current, &heads, &rels);
            let tail_log_probs = activation::log_softmax(tail_scores, 1);

            // Head prediction: score all entities for (?, r, t).
            let head_scores = score_1n_heads(&current, &rels, &tails);
            let head_log_probs = activation::log_softmax(head_scores, 1);

            // 1vsAll CE: gather the correct entity's log-prob.
            let tail_ids = tails.clone().unsqueeze_dim(1); // [bs, 1]
            let t_nll = tail_log_probs
                .clone()
                .gather(1, tail_ids)
                .squeeze::<1>()
                .neg()
                .mean();

            let head_ids = heads.clone().unsqueeze_dim(1); // [bs, 1]
            let h_nll = head_log_probs
                .clone()
                .gather(1, head_ids)
                .squeeze::<1>()
                .neg()
                .mean();

            let nll = (t_nll + h_nll) / 2.0;

            let loss = if eps > 0.0 {
                let tail_uniform = tail_log_probs.mean().neg();
                let head_uniform = head_log_probs.mean().neg();
                let uniform = (tail_uniform + head_uniform) / 2.0;
                nll * (1.0 - eps) + uniform * eps
            } else {
                nll
            };

            // Extract loss value from inner (non-autodiff) tensor to avoid
            // consuming the computation graph before backward().
            let loss_val: f32 = loss.clone().inner().into_scalar().to_f32();
            let grads = GradientsParams::from_grads(loss.backward(), &current);

            if loss_val.is_finite() {
                model = optim.step(config.lr, current, grads);
            }

            epoch_loss += loss_val as f64;
            n_batches += 1;
        }

        let avg_loss = (epoch_loss / n_batches as f64) as f32;
        losses.push(avg_loss);

        if config.log_interval > 0 && (epoch + 1) % config.log_interval == 0 {
            eprintln!(
                "epoch {:>4} | loss {:.4} | {:.1}s",
                epoch + 1,
                avg_loss,
                epoch_start.elapsed().as_secs_f32(),
            );
        }
    }

    // Extract embeddings to CPU.
    let dim = config.dim;
    let extract = |re: &Param<Tensor<B, 2>>, im: &Param<Tensor<B, 2>>| -> Vec<Vec<f32>> {
        let re_data: Vec<f32> = re.val().into_data().to_vec().unwrap();
        let im_data: Vec<f32> = im.val().into_data().to_vec().unwrap();
        let n = re_data.len() / dim;
        (0..n)
            .map(|i| {
                let mut v = Vec::with_capacity(dim * 2);
                v.extend_from_slice(&re_data[i * dim..(i + 1) * dim]);
                v.extend_from_slice(&im_data[i * dim..(i + 1) * dim]);
                v
            })
            .collect()
    };

    BurnTrainResult {
        entity_vecs: extract(&model.entity_re, &model.entity_im),
        relation_vecs: extract(&model.relation_re, &model.relation_im),
        dim,
        losses,
    }
}

// ---------------------------------------------------------------------------
// Generic KGE trainer: all four models with 1-N (1vsAll) scoring
// ---------------------------------------------------------------------------

/// KGE model family for the generic burn trainer.
///
/// Mirrors the candle `crate::train::ModelType` scoring, but is defined here so
/// the burn path does not depend on the candle-gated `train` module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BurnModelType {
    /// `-||h + r - t||^2` (translation, Bordes et al. 2013).
    TransE,
    /// `-||h o r - t||^2` with relation as a per-dim phase rotation (Sun et al. 2019).
    RotatE,
    /// `Re(<h, r, conj(t)>)` (complex bilinear, Trouillon et al. 2016).
    ComplEx,
    /// `<h, r, t>` (real bilinear, Yang et al. 2015).
    DistMult,
}

impl BurnModelType {
    /// Floats stored per entity (`2*dim` for complex models, `dim` otherwise).
    fn entity_dim(self, dim: usize) -> usize {
        match self {
            BurnModelType::ComplEx | BurnModelType::RotatE => 2 * dim,
            BurnModelType::TransE | BurnModelType::DistMult => dim,
        }
    }
    /// Floats stored per relation (`2*dim` for ComplEx, `dim` otherwise).
    fn relation_dim(self, dim: usize) -> usize {
        match self {
            BurnModelType::ComplEx => 2 * dim,
            _ => dim,
        }
    }
}

/// Generic KGE model: entity and relation embedding tables.
#[derive(Module, Debug)]
pub struct BurnKge<B: Backend> {
    entity: Param<Tensor<B, 2>>,
    relation: Param<Tensor<B, 2>>,
}

/// Result of generic KGE training.
pub struct BurnKgeResult {
    /// Entity embeddings, one row per entity.
    pub entity_vecs: Vec<Vec<f32>>,
    /// Relation embeddings, one row per relation.
    pub relation_vecs: Vec<Vec<f32>>,
    /// (Complex) dimension.
    pub dim: usize,
    /// Model family.
    pub model_type: BurnModelType,
    /// Loss per epoch.
    pub losses: Vec<f32>,
}

impl BurnKgeResult {
    /// Build the matching CPU scorer for evaluation.
    pub fn to_scorer(&self) -> Box<dyn crate::Scorer + Sync> {
        let e = self.entity_vecs.clone();
        let r = self.relation_vecs.clone();
        match self.model_type {
            BurnModelType::TransE => Box::new(crate::TransE::from_vecs(e, r, self.dim)),
            BurnModelType::DistMult => Box::new(crate::DistMult::from_vecs(e, r, self.dim)),
            BurnModelType::ComplEx => Box::new(crate::ComplEx::from_vecs(e, r, self.dim)),
            BurnModelType::RotatE => Box::new(crate::RotatE::from_vecs(e, r, self.dim, 12.0)),
        }
    }
}

/// Split a `[rows, 2*dim]` complex tensor into `(re, im)` halves.
fn re_im<B: Backend>(t: Tensor<B, 2>, dim: usize) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let rows = t.dims()[0];
    (
        t.clone().slice([0..rows, 0..dim]),
        t.slice([0..rows, dim..2 * dim]),
    )
}

/// `-||hr - e||^2` for every entity `e`, via the GEMM expansion of the squared
/// distance. `hr` is `[B, d]`, `ent` is `[E, d]`; returns `[B, E]`.
fn neg_sq_dist<B: Backend>(hr: Tensor<B, 2>, ent: Tensor<B, 2>) -> Tensor<B, 2> {
    let hr_sq = hr.clone().powf_scalar(2.0).sum_dim(1); // [B, 1]
    let ent_sq = ent.clone().powf_scalar(2.0).sum_dim(1).transpose(); // [1, E]
    let cross = hr.matmul(ent.transpose()); // [B, E]
    (hr_sq + ent_sq - cross.mul_scalar(2.0)).neg()
}

/// Score all entities as tails for `(h, r, ?)`. Higher = more likely. `[B, E]`.
fn score_1n_kge<B: Backend>(
    model: &BurnKge<B>,
    mt: BurnModelType,
    dim: usize,
    heads: &Tensor<B, 1, Int>,
    rels: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let h = model.entity.val().select(0, heads.clone());
    let r = model.relation.val().select(0, rels.clone());
    let ent = model.entity.val();
    match mt {
        BurnModelType::TransE => neg_sq_dist(h + r, ent),
        BurnModelType::DistMult => (h * r).matmul(ent.transpose()),
        BurnModelType::ComplEx => {
            let (h_re, h_im) = re_im(h, dim);
            let (r_re, r_im) = re_im(r, dim);
            let hr_re = h_re.clone() * r_re.clone() - h_im.clone() * r_im.clone();
            let hr_im = h_re * r_im + h_im * r_re;
            let (e_re, e_im) = re_im(ent, dim);
            hr_re.matmul(e_re.transpose()) + hr_im.matmul(e_im.transpose())
        }
        BurnModelType::RotatE => {
            let (h_re, h_im) = re_im(h, dim);
            let cos = r.clone().cos();
            let sin = r.sin();
            let hr_re = h_re.clone() * cos.clone() - h_im.clone() * sin.clone();
            let hr_im = h_re * sin + h_im * cos;
            neg_sq_dist(Tensor::cat(vec![hr_re, hr_im], 1), ent)
        }
    }
}

/// Score all entities as heads for `(?, r, t)`. Higher = more likely. `[B, E]`.
fn score_1n_heads_kge<B: Backend>(
    model: &BurnKge<B>,
    mt: BurnModelType,
    dim: usize,
    rels: &Tensor<B, 1, Int>,
    tails: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let r = model.relation.val().select(0, rels.clone());
    let t = model.entity.val().select(0, tails.clone());
    let ent = model.entity.val();
    match mt {
        BurnModelType::TransE => neg_sq_dist(t - r, ent),
        BurnModelType::DistMult => (r * t).matmul(ent.transpose()),
        BurnModelType::ComplEx => {
            let (r_re, r_im) = re_im(r, dim);
            let (t_re, t_im) = re_im(t, dim);
            let rc_re = r_re.clone() * t_re.clone() + r_im.clone() * t_im.clone();
            let rc_im = r_im * t_re - r_re * t_im;
            let (e_re, e_im) = re_im(ent, dim);
            rc_re.matmul(e_re.transpose()) + rc_im.matmul(e_im.transpose())
        }
        BurnModelType::RotatE => {
            // |r| = 1, so ||e o r - t|| = ||e - t o conj(r)||; target = t o conj(r).
            let (t_re, t_im) = re_im(t, dim);
            let cos = r.clone().cos();
            let sin = r.sin();
            let tr_re = t_re.clone() * cos.clone() + t_im.clone() * sin.clone();
            let tr_im = t_im * cos - t_re * sin;
            neg_sq_dist(Tensor::cat(vec![tr_re, tr_im], 1), ent)
        }
    }
}

/// Train any of the four KGE models with 1-N (1vsAll) scoring.
///
/// Generic counterpart of [`train_complex`]; the `BurnModelType` selects the
/// scoring function. Returns embeddings, the model family, and per-epoch loss.
pub fn train_kge<B: AutodiffBackend>(
    train_triples: &[crate::dataset::TripleIds],
    num_entities: usize,
    num_relations: usize,
    model_type: BurnModelType,
    config: &BurnTrainConfig,
    device: &B::Device,
) -> BurnKgeResult {
    let ent_dim = model_type.entity_dim(config.dim);
    let rel_dim = model_type.relation_dim(config.dim);
    let mk = |rows, cols| {
        Param::initialized(
            ParamId::new(),
            Tensor::<B, 2>::random(
                [rows, cols],
                burn::tensor::Distribution::Normal(0.0, config.init_scale),
                device,
            )
            .require_grad(),
        )
    };
    let mut model = BurnKge {
        entity: mk(num_entities, ent_dim),
        relation: mk(num_relations, rel_dim),
    };
    let mut optim = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .init::<B, BurnKge<B>>();

    let n_triples = train_triples.len();
    let batch_size = config.batch_size.min(n_triples).max(1);
    let eps = config.label_smoothing;
    let mut losses = Vec::with_capacity(config.epochs);
    let mut indices: Vec<usize> = (0..n_triples).collect();

    for _epoch in 0..config.epochs {
        {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::rng());
        }
        // Accumulate the detached loss on-device and read it once per epoch.
        // Reading the loss every batch (into_scalar) forces a CPU<->GPU sync that
        // stalls the pipeline -- the dominant per-batch cost on wgpu/Metal.
        let mut epoch_loss_acc: Option<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = None;
        let mut n_batches = 0u32;
        let mut offset = 0;
        while offset < n_triples {
            let end = (offset + batch_size).min(n_triples);
            let batch_idx = &indices[offset..end];
            let bs = batch_idx.len();
            offset = end;

            let hd: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].head as i64)
                .collect();
            let rd: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].relation as i64)
                .collect();
            let tdv: Vec<i64> = batch_idx
                .iter()
                .map(|&i| train_triples[i].tail as i64)
                .collect();
            let heads =
                Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(hd, [bs]), device);
            let rels =
                Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(rd, [bs]), device);
            let tails =
                Tensor::<B, 1, Int>::from_data(burn::tensor::TensorData::new(tdv, [bs]), device);

            let current = model.clone();
            let tail_scores = score_1n_kge(&current, model_type, config.dim, &heads, &rels);
            let tail_lp = activation::log_softmax(tail_scores, 1);
            let head_scores = score_1n_heads_kge(&current, model_type, config.dim, &rels, &tails);
            let head_lp = activation::log_softmax(head_scores, 1);

            let t_nll = tail_lp
                .clone()
                .gather(1, tails.clone().unsqueeze_dim(1))
                .squeeze::<1>()
                .neg()
                .mean();
            let h_nll = head_lp
                .clone()
                .gather(1, heads.clone().unsqueeze_dim(1))
                .squeeze::<1>()
                .neg()
                .mean();
            let nll = (t_nll + h_nll) / 2.0;
            let loss = if eps > 0.0 {
                let uniform = (tail_lp.mean().neg() + head_lp.mean().neg()) / 2.0;
                nll * (1.0 - eps) + uniform * eps
            } else {
                nll
            };

            let loss_inner = loss.clone().inner();
            let grads = GradientsParams::from_grads(loss.backward(), &current);
            model = optim.step(config.lr, current, grads);
            epoch_loss_acc = Some(match epoch_loss_acc {
                Some(acc) => acc + loss_inner,
                None => loss_inner,
            });
            n_batches += 1;
        }
        let avg = match epoch_loss_acc {
            Some(acc) => acc.into_scalar().to_f32() / n_batches.max(1) as f32,
            None => 0.0,
        };
        losses.push(avg);
    }

    let extract = |p: &Param<Tensor<B, 2>>, cols: usize| -> Vec<Vec<f32>> {
        let data: Vec<f32> = p.val().into_data().to_vec().unwrap();
        data.chunks(cols).map(<[f32]>::to_vec).collect()
    };
    BurnKgeResult {
        entity_vecs: extract(&model.entity, ent_dim),
        relation_vecs: extract(&model.relation, rel_dim),
        dim: config.dim,
        model_type,
        losses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::TripleIds;
    use crate::Scorer;

    fn tid(h: usize, r: usize, t: usize) -> TripleIds {
        TripleIds::new(h, r, t)
    }

    #[cfg(feature = "burn-ndarray")]
    type TestBackend = burn::backend::Autodiff<burn_ndarray::NdArray>;

    #[cfg(feature = "burn-ndarray")]
    fn test_device() -> <TestBackend as Backend>::Device {
        burn_ndarray::NdArrayDevice::Cpu
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_complex_smoke() {
        let triples = vec![tid(0, 0, 1), tid(1, 0, 2), tid(2, 1, 0), tid(0, 1, 2)];
        let config = BurnTrainConfig {
            dim: 8,
            epochs: 10,
            batch_size: 4,
            ..BurnTrainConfig::default()
        };
        let result = train_complex::<TestBackend>(&triples, 3, 2, &config, &test_device());
        assert_eq!(result.losses.len(), 10);
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let model = result.to_complex();
        assert_eq!(model.num_entities(), 3);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_complex_loss_decreases() {
        let triples: Vec<_> = (0..20).map(|i| tid(i % 5, i % 2, (i + 1) % 5)).collect();
        let config = BurnTrainConfig {
            dim: 16,
            epochs: 30,
            batch_size: 10,
            lr: 0.001,
            ..BurnTrainConfig::default()
        };
        let result = train_complex::<TestBackend>(&triples, 5, 2, &config, &test_device());
        let first = result.losses[0];
        let last = *result.losses.last().unwrap();
        assert!(
            last < first,
            "Burn ComplEx loss should decrease: {first} -> {last}"
        );
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_complex_achieves_nonzero_mrr() {
        let triples = vec![tid(0, 0, 1), tid(1, 0, 2), tid(2, 0, 3), tid(3, 0, 4)];
        let config = BurnTrainConfig {
            dim: 32,
            epochs: 200,
            batch_size: 4,
            lr: 0.001,
            ..BurnTrainConfig::default()
        };
        let result = train_complex::<TestBackend>(&triples, 5, 1, &config, &test_device());
        let model = result.to_complex();

        let ds = crate::dataset::Dataset::new(
            triples
                .iter()
                .map(|t| {
                    crate::dataset::Triple::new(
                        t.head.to_string(),
                        t.relation.to_string(),
                        t.tail.to_string(),
                    )
                })
                .collect(),
            Vec::new(),
            Vec::new(),
        )
        .into_interned();
        let filter = crate::dataset::FilterIndex::from_dataset(&ds);
        let metrics = crate::eval::evaluate_link_prediction(&model, &triples, &filter, 5);
        assert!(
            metrics.mrr > 0.3,
            "Burn ComplEx should achieve MRR > 0.3, got {:.4}",
            metrics.mrr
        );
    }

    #[cfg(feature = "burn-ndarray")]
    fn kge_loss_decreases(mt: BurnModelType) {
        let triples: Vec<_> = (0..20).map(|i| tid(i % 5, i % 2, (i + 1) % 5)).collect();
        let config = BurnTrainConfig {
            dim: 16,
            epochs: 40,
            batch_size: 10,
            lr: 0.005,
            ..BurnTrainConfig::default()
        };
        let result = train_kge::<TestBackend>(&triples, 5, 2, mt, &config, &test_device());
        assert_eq!(result.losses.len(), 40);
        assert!(
            result.losses.iter().all(|l| l.is_finite()),
            "{mt:?} produced a non-finite loss"
        );
        let (first, last) = (result.losses[0], *result.losses.last().unwrap());
        assert!(
            last < first,
            "{mt:?} loss should decrease: {first} -> {last}"
        );
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_transe_loss_decreases() {
        kge_loss_decreases(BurnModelType::TransE);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_distmult_loss_decreases() {
        kge_loss_decreases(BurnModelType::DistMult);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_rotate_loss_decreases() {
        kge_loss_decreases(BurnModelType::RotatE);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_distmult_achieves_nonzero_mrr() {
        let triples = vec![tid(0, 0, 1), tid(1, 0, 2), tid(2, 0, 3), tid(3, 0, 4)];
        let config = BurnTrainConfig {
            dim: 32,
            epochs: 200,
            batch_size: 4,
            lr: 0.01,
            ..BurnTrainConfig::default()
        };
        let result = train_kge::<TestBackend>(
            &triples,
            5,
            1,
            BurnModelType::DistMult,
            &config,
            &test_device(),
        );
        let model = crate::DistMult::from_vecs(
            result.entity_vecs.clone(),
            result.relation_vecs.clone(),
            result.dim,
        );
        let ds = crate::dataset::Dataset::new(
            triples
                .iter()
                .map(|t| {
                    crate::dataset::Triple::new(
                        t.head.to_string(),
                        t.relation.to_string(),
                        t.tail.to_string(),
                    )
                })
                .collect(),
            Vec::new(),
            Vec::new(),
        )
        .into_interned();
        let filter = crate::dataset::FilterIndex::from_dataset(&ds);
        let metrics = crate::eval::evaluate_link_prediction(&model, &triples, &filter, 5);
        assert!(
            metrics.mrr > 0.3,
            "Burn DistMult should achieve MRR > 0.3, got {:.4}",
            metrics.mrr
        );
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_kge_to_scorer_builds_all_models() {
        let triples = vec![tid(0, 0, 1), tid(1, 0, 2)];
        let config = BurnTrainConfig {
            dim: 8,
            epochs: 5,
            batch_size: 2,
            ..BurnTrainConfig::default()
        };
        for mt in [
            BurnModelType::TransE,
            BurnModelType::RotatE,
            BurnModelType::ComplEx,
            BurnModelType::DistMult,
        ] {
            let result = train_kge::<TestBackend>(&triples, 3, 1, mt, &config, &test_device());
            assert!(result.losses.iter().all(|l| l.is_finite()), "{mt:?}");
            let scorer = result.to_scorer();
            assert_eq!(scorer.num_entities(), 3, "{mt:?}");
        }
    }
}
