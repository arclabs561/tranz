//! Burn-based training for KGE models.
//!
//! Trains all four point-embedding models on CPU (ndarray + rayon) or GPU
//! (WGPU/Metal/Vulkan) backends via burn's backend system.
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
    /// Weighted nuclear-3 (Ω³) coefficient, Lacroix et al. (ICML 2018):
    /// per sampled triple, `(λ/3)(‖h‖₃³ + ‖r‖₃³ + ‖t‖₃³)` on the sampled
    /// rows (complex modulus per component for ComplEx). Applied by
    /// [`train_kge`] for the CP-family models (ComplEx, DistMult); ignored
    /// for the distance models (TransE, RotatE), where it is not the
    /// canonical regularizer. 0 = disabled. Pair with `init_scale` ~1e-2:
    /// at tiny init the origin is a fixed point of the multilinear score
    /// and the N3 pull wins.
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

    // Re(h * rc) = h_re * rc_re - h_im * rc_im (h is un-conjugated in the
    // head role, so the imaginary product is subtracted). Against all
    // entities: rc_re @ e_re^T - rc_im @ e_im^T.
    let e_re = model.entity_re.val();
    let e_im = model.entity_im.val();
    rc_re.matmul(e_re.transpose()) - rc_im.matmul(e_im.transpose())
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
/// Selects the scoring function used by [`train_kge`] and the CPU scorer built
/// by [`BurnKgeResult::to_scorer`].
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
            // Re(h * rc): minus on the imaginary product (h un-conjugated).
            rc_re.matmul(e_re.transpose()) - rc_im.matmul(e_im.transpose())
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
            let mut loss = if eps > 0.0 {
                let uniform = (tail_lp.mean().neg() + head_lp.mean().neg()) / 2.0;
                nll * (1.0 - eps) + uniform * eps
            } else {
                nll
            };
            if config.n3_reg > 0.0
                && matches!(model_type, BurnModelType::ComplEx | BurnModelType::DistMult)
            {
                let h = current.entity.val().select(0, heads.clone());
                let t = current.entity.val().select(0, tails.clone());
                let r = current.relation.val().select(0, rels.clone());
                let cube = |x: Tensor<B, 2>, complex: bool| {
                    if complex {
                        let d = x.dims()[1] / 2;
                        let (re, im) = re_im(x, d);
                        (re.powf_scalar(2.0) + im.powf_scalar(2.0))
                            .add_scalar(1e-12)
                            .sqrt()
                            .powf_scalar(3.0)
                            .sum_dim(1)
                    } else {
                        x.abs().powf_scalar(3.0).sum_dim(1)
                    }
                };
                let complex = model_type == BurnModelType::ComplEx;
                let omega3 = (cube(h, complex) + cube(t, complex) + cube(r, complex)).mean();
                loss = loss + omega3.mul_scalar(config.n3_reg / 3.0);
            }

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
    fn burn_complex_scores_match_cpu_reference() {
        // Parity contract between the burn scoring paths and the CPU
        // reference: identical weights must produce identical scores (the
        // CPU Scorer returns energies, lower = better, so burn + cpu ~= 0).
        // Guards the head-role conjugation: Re(h * r * conj(t)) carries a
        // minus on the imaginary product when scoring over heads.
        let (ne, nr, dim) = (5usize, 2usize, 4usize);
        let val = |i: usize| ((i * 37 + 11) % 19) as f32 / 19.0 - 0.5;
        let ent_rows: Vec<Vec<f32>> = (0..ne)
            .map(|e| (0..dim * 2).map(|j| val(e * 31 + j)).collect())
            .collect();
        let rel_rows: Vec<Vec<f32>> = (0..nr)
            .map(|r| (0..dim * 2).map(|j| val(r * 53 + j + 7)).collect())
            .collect();
        let cpu = crate::ComplEx::from_vecs(ent_rows.clone(), rel_rows.clone(), dim);

        let device = test_device();
        let param = |rows: &[Vec<f32>], lo: usize, hi: usize| {
            let flat: Vec<f32> = rows
                .iter()
                .flat_map(|r| r[lo..hi].iter().copied())
                .collect();
            Param::initialized(
                ParamId::new(),
                Tensor::<TestBackend, 2>::from_data(
                    burn::tensor::TensorData::new(flat, [rows.len(), hi - lo]),
                    &device,
                ),
            )
        };
        let model = BurnComplEx::<TestBackend> {
            entity_re: param(&ent_rows, 0, dim),
            entity_im: param(&ent_rows, dim, dim * 2),
            relation_re: param(&rel_rows, 0, dim),
            relation_im: param(&rel_rows, dim, dim * 2),
        };
        // The generic KGE model shares the concatenated [re.., im..] layout.
        let kge = BurnKge::<TestBackend> {
            entity: param(&ent_rows, 0, dim * 2),
            relation: param(&rel_rows, 0, dim * 2),
        };

        let scores =
            |t: Tensor<TestBackend, 2>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
        for r in 0..nr {
            for x in 0..ne {
                let ids = |i: usize| {
                    Tensor::<TestBackend, 1, Int>::from_data(
                        burn::tensor::TensorData::new(vec![i as i64], [1]),
                        &device,
                    )
                };
                let (rels, xs) = (ids(r), ids(x));
                let mt = BurnModelType::ComplEx;
                let cases = [
                    (
                        "heads",
                        scores(score_1n_heads(&model, &rels, &xs)),
                        cpu.score_all_heads(r, x),
                    ),
                    (
                        "tails",
                        scores(score_1n(&model, &xs, &rels)),
                        cpu.score_all_tails(x, r),
                    ),
                    (
                        "kge heads",
                        scores(score_1n_heads_kge(&kge, mt, dim, &rels, &xs)),
                        cpu.score_all_heads(r, x),
                    ),
                    (
                        "kge tails",
                        scores(score_1n_kge(&kge, mt, dim, &xs, &rels)),
                        cpu.score_all_tails(x, r),
                    ),
                ];
                for (name, burn_scores, cpu_energies) in cases {
                    assert_eq!(burn_scores.len(), ne);
                    for (e, (b, c)) in burn_scores.iter().zip(cpu_energies.iter()).enumerate() {
                        assert!(
                            (b + c).abs() < 1e-4,
                            "{name} mismatch at rel={r} x={x} entity={e}: burn={b} cpu={}",
                            -c
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_distmult_scores_match_cpu_reference() {
        // Exact parity between the generic KGE burn scorer (DistMult path) and
        // the CPU reference: identical weights must produce identical scores.
        // DistMult's CPU Scorer returns -score_triple (energy, lower = better),
        // so burn + cpu ~= 0, mirroring the ComplEx contract. DistMult has no
        // model-specific burn scorer, so only the generic score_1n_kge path
        // applies.
        let (ne, nr, dim) = (5usize, 2usize, 4usize);
        let val = |i: usize| ((i * 37 + 11) % 19) as f32 / 19.0 - 0.5;
        let ent_rows: Vec<Vec<f32>> = (0..ne)
            .map(|e| (0..dim).map(|j| val(e * 31 + j)).collect())
            .collect();
        let rel_rows: Vec<Vec<f32>> = (0..nr)
            .map(|r| (0..dim).map(|j| val(r * 53 + j + 7)).collect())
            .collect();
        let cpu = crate::DistMult::from_vecs(ent_rows.clone(), rel_rows.clone(), dim);

        let device = test_device();
        let param = |rows: &[Vec<f32>]| {
            let cols = rows[0].len();
            let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
            Param::initialized(
                ParamId::new(),
                Tensor::<TestBackend, 2>::from_data(
                    burn::tensor::TensorData::new(flat, [rows.len(), cols]),
                    &device,
                ),
            )
        };
        let kge = BurnKge::<TestBackend> {
            entity: param(&ent_rows),
            relation: param(&rel_rows),
        };

        let scores =
            |t: Tensor<TestBackend, 2>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
        let mt = BurnModelType::DistMult;
        for r in 0..nr {
            for x in 0..ne {
                let ids = |i: usize| {
                    Tensor::<TestBackend, 1, Int>::from_data(
                        burn::tensor::TensorData::new(vec![i as i64], [1]),
                        &device,
                    )
                };
                let (rels, xs) = (ids(r), ids(x));
                let cases = [
                    (
                        "kge heads",
                        scores(score_1n_heads_kge(&kge, mt, dim, &rels, &xs)),
                        cpu.score_all_heads(r, x),
                    ),
                    (
                        "kge tails",
                        scores(score_1n_kge(&kge, mt, dim, &xs, &rels)),
                        cpu.score_all_tails(x, r),
                    ),
                ];
                for (name, burn_scores, cpu_energies) in cases {
                    assert_eq!(burn_scores.len(), ne);
                    for (e, (b, c)) in burn_scores.iter().zip(cpu_energies.iter()).enumerate() {
                        assert!(
                            (b + c).abs() < 1e-4,
                            "{name} mismatch at rel={r} x={x} entity={e}: burn={b} cpu={}",
                            -c
                        );
                    }
                }
            }
        }
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

    #[cfg(feature = "burn-ndarray")]
    fn kge_ranks_match_cpu_reference(mt: BurnModelType) {
        // TransE and RotatE burn scorers use squared L2 distance while the CPU
        // reference uses plain L2 -- a monotonic transform, so exact score
        // parity is impossible but the induced candidate ORDERING is identical.
        // Rank parity catches a sign or conjugation error without requiring
        // score equality.
        let (ne, nr, dim) = (6usize, 2usize, 4usize);
        // Low-discrepancy deterministic weights, well-spread so distances are
        // separated enough that f32 GEMM noise cannot flip a near-tie.
        let val = |i: usize| {
            let x = ((i as f64) * 0.618_033_988_749_895).fract();
            (2.0 * x - 1.0) as f32
        };
        let (ent_w, rel_w) = match mt {
            BurnModelType::TransE => (dim, dim),
            BurnModelType::RotatE => (2 * dim, dim),
            other => panic!("rank-parity helper is for distance models, not {other:?}"),
        };
        let ent_rows: Vec<Vec<f32>> = (0..ne)
            .map(|e| (0..ent_w).map(|j| val(e * 31 + j + 1)).collect())
            .collect();
        let rel_rows: Vec<Vec<f32>> = (0..nr)
            .map(|r| (0..rel_w).map(|j| val(r * 53 + j + 7)).collect())
            .collect();

        let cpu: Box<dyn Scorer + Sync> = match mt {
            BurnModelType::TransE => Box::new(crate::TransE::from_vecs(
                ent_rows.clone(),
                rel_rows.clone(),
                dim,
            )),
            BurnModelType::RotatE => Box::new(crate::RotatE::from_vecs(
                ent_rows.clone(),
                rel_rows.clone(),
                dim,
                12.0,
            )),
            other => panic!("rank-parity helper is for distance models, not {other:?}"),
        };

        let device = test_device();
        let param = |rows: &[Vec<f32>]| {
            let cols = rows[0].len();
            let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
            Param::initialized(
                ParamId::new(),
                Tensor::<TestBackend, 2>::from_data(
                    burn::tensor::TensorData::new(flat, [rows.len(), cols]),
                    &device,
                ),
            )
        };
        let kge = BurnKge::<TestBackend> {
            entity: param(&ent_rows),
            relation: param(&rel_rows),
        };

        let scores =
            |t: Tensor<TestBackend, 2>| -> Vec<f32> { t.into_data().to_vec::<f32>().unwrap() };
        // Best-to-worst entity order. Burn: higher score first. CPU: lower
        // distance first. Ties broken by ascending entity index in both.
        let rank_by = |v: &[f32], higher_is_better: bool| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| {
                let ord = if higher_is_better {
                    v[b].total_cmp(&v[a])
                } else {
                    v[a].total_cmp(&v[b])
                };
                ord.then(a.cmp(&b))
            });
            idx
        };

        for r in 0..nr {
            for x in 0..ne {
                let ids = |i: usize| {
                    Tensor::<TestBackend, 1, Int>::from_data(
                        burn::tensor::TensorData::new(vec![i as i64], [1]),
                        &device,
                    )
                };
                let (rels, xs) = (ids(r), ids(x));

                let burn_tails = scores(score_1n_kge(&kge, mt, dim, &xs, &rels));
                let cpu_tails = cpu.score_all_tails(x, r);
                assert_eq!(
                    rank_by(&burn_tails, true),
                    rank_by(&cpu_tails, false),
                    "{mt:?} tail ordering at rel={r} x={x}: burn={burn_tails:?} cpu={cpu_tails:?}"
                );

                let burn_heads = scores(score_1n_heads_kge(&kge, mt, dim, &rels, &xs));
                let cpu_heads = cpu.score_all_heads(r, x);
                assert_eq!(
                    rank_by(&burn_heads, true),
                    rank_by(&cpu_heads, false),
                    "{mt:?} head ordering at rel={r} x={x}: burn={burn_heads:?} cpu={cpu_heads:?}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_transe_ranks_match_cpu_reference() {
        kge_ranks_match_cpu_reference(BurnModelType::TransE);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_rotate_ranks_match_cpu_reference() {
        kge_ranks_match_cpu_reference(BurnModelType::RotatE);
    }

    #[cfg(feature = "burn-ndarray")]
    fn kge_achieves_nonzero_mrr(mt: BurnModelType) {
        // The chain 0->1->2->3->4 under a single relation is trivially learnable
        // by a translation/rotation model, so a trained burn scorer should rank
        // the true entity near the top (MRR > 0.3). Adapted from
        // burn_distmult_achieves_nonzero_mrr for the generic trainer.
        let triples = vec![tid(0, 0, 1), tid(1, 0, 2), tid(2, 0, 3), tid(3, 0, 4)];
        let config = BurnTrainConfig {
            dim: 32,
            epochs: 300,
            batch_size: 4,
            lr: 0.01,
            ..BurnTrainConfig::default()
        };
        let result = train_kge::<TestBackend>(&triples, 5, 1, mt, &config, &test_device());
        let scorer = result.to_scorer();

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
        let metrics = crate::eval::evaluate_link_prediction(scorer.as_ref(), &triples, &filter, 5);
        assert!(
            metrics.mrr > 0.3,
            "Burn {mt:?} should achieve MRR > 0.3, got {:.4}",
            metrics.mrr
        );
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_transe_achieves_nonzero_mrr() {
        kge_achieves_nonzero_mrr(BurnModelType::TransE);
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_rotate_achieves_nonzero_mrr() {
        kge_achieves_nonzero_mrr(BurnModelType::RotatE);
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
