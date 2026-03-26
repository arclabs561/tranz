//! Training loop for KGE models via candle.
//!
//! Implements:
//! - Negative sampling with configurable corruption strategy
//! - Self-adversarial negative sampling (SANS) weighting
//! - Log-sigmoid loss with margin
//! - N3 regularization (nuclear 3-norm, for ComplEx/DistMult)
//! - AdamW optimizer
//!
//! ## Training protocol
//!
//! For each batch of positive triples `(h, r, t)`:
//! 1. Sample `k` negative triples by corrupting head or tail.
//! 2. Score positives and negatives.
//! 3. Weight negatives by SANS: `p_i = softmax(alpha * score_i)` (detached).
//! 4. Loss = `-log(sigma(gamma - score_pos)) - sum_i p_i * log(sigma(score_neg_i - gamma))`.
//! 5. Optionally add N3 regularization.
//! 6. Backward + optimizer step.

use candle_core::{DType, Device, IndexOp, Result, Tensor, Var, D};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

/// Which model architecture to train.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// TransE: `||h + r - t||`.
    TransE,
    /// RotatE: `||h * r - t||` in complex space.
    RotatE,
    /// ComplEx: `Re(h * r * conj(t))`.
    ComplEx,
    /// DistMult: `sum(h * r * t)`.
    DistMult,
}

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Model type.
    pub model_type: ModelType,
    /// Embedding dimension (complex dim for RotatE/ComplEx).
    pub dim: usize,
    /// Number of negative samples per positive (ignored in 1-N mode).
    pub num_negatives: usize,
    /// Use 1-N scoring: score all entities per (h,r) query with BCE loss.
    /// Much faster convergence (5-10x fewer epochs) than negative sampling.
    /// Requires more memory per batch: `batch_size * num_entities * 4` bytes.
    pub one_to_n: bool,
    /// Label smoothing epsilon for 1-N mode. 0 = no smoothing.
    /// Recommended: 0.1. Replaces hard 0/1 targets with (eps, 1-eps).
    pub label_smoothing: f32,
    /// Margin gamma for the loss function (used in negative sampling mode).
    pub gamma: f32,
    /// Norm for distance-based models (TransE, RotatE). 1 = L1, 2 = L2.
    /// The RotatE reference implementation uses L1 for TransE.
    pub distance_norm: u32,
    /// Apply subsampling weights based on entity frequency.
    /// Downweights triples involving high-frequency entities.
    /// Helps at convergence but can hurt during early training.
    pub subsampling: bool,
    /// SANS adversarial temperature. 0 = uniform weighting.
    pub adversarial_temperature: f32,
    /// Learning rate.
    pub lr: f64,
    /// Dropout rate on entity/relation embeddings. 0 = no dropout.
    /// Recommended: 0.1-0.2 (Ruffinelli et al. 2020).
    pub embedding_dropout: f32,
    /// N3 regularization coefficient. 0 = disabled.
    pub n3_reg: f32,
    /// Batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Normalize entity embeddings to unit L2 norm after each step.
    /// Standard for TransE (Bordes et al., 2013). Disabled by default.
    pub normalize_entities: bool,
    /// Linear warmup epochs. LR ramps from 0 to `lr` over this many epochs.
    /// 0 = no warmup.
    pub warmup_epochs: usize,
    /// Print loss to stderr every N epochs. 0 = silent.
    pub log_interval: usize,
    /// Evaluate on validation set every N epochs. 0 = no validation.
    pub eval_interval: usize,
    /// Stop if validation MRR doesn't improve for this many eval cycles.
    /// Only used when `eval_interval > 0`.
    pub patience: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::TransE,
            dim: 200,
            num_negatives: 256,
            one_to_n: false,
            label_smoothing: 0.0,
            gamma: 12.0,
            distance_norm: 1,
            subsampling: false,
            adversarial_temperature: 1.0,
            lr: 0.001,
            embedding_dropout: 0.0,
            n3_reg: 0.0,
            batch_size: 512,
            epochs: 1000,
            normalize_entities: false,
            warmup_epochs: 0,
            log_interval: 0,
            eval_interval: 0,
            patience: 5,
        }
    }
}

/// Trained model with candle tensors.
///
/// Holds the embedding `Var`s for gradient-based training, and can
/// extract to CPU `Vec<Vec<f32>>` for evaluation via the `Scorer` trait.
pub struct TrainableModel {
    entity_embeddings: Var,
    relation_embeddings: Var,
    model_type: ModelType,
    dim: usize,
    gamma: f32,
    distance_norm: u32,
    embedding_dropout: f32,
    device: Device,
}

impl TrainableModel {
    /// Initialize a new trainable model.
    pub fn new(
        num_entities: usize,
        num_relations: usize,
        config: &TrainConfig,
        device: &Device,
    ) -> Result<Self> {
        let dim = config.dim;
        let gamma = config.gamma;

        let (entity_embeddings, relation_embeddings) = match config.model_type {
            ModelType::TransE => {
                let scale = 6.0 / (dim as f64).sqrt();
                let ent =
                    Var::rand_f64(0.0 - scale, scale, (num_entities, dim), DType::F32, device)?;
                let rel =
                    Var::rand_f64(0.0 - scale, scale, (num_relations, dim), DType::F32, device)?;
                (ent, rel)
            }
            ModelType::RotatE => {
                // Entities: interleaved re/im, so dim*2 columns.
                let ent_scale = gamma as f64 / (dim as f64).sqrt();
                let ent = Var::rand_f64(
                    -ent_scale,
                    ent_scale,
                    (num_entities, dim * 2),
                    DType::F32,
                    device,
                )?;
                // Relations: angles in [-pi, pi].
                let rel = Var::rand_f64(
                    -std::f64::consts::PI,
                    std::f64::consts::PI,
                    (num_relations, dim),
                    DType::F32,
                    device,
                )?;
                (ent, rel)
            }
            ModelType::ComplEx | ModelType::DistMult => {
                let scale = (6.0 / dim as f64).sqrt();
                let ent_cols = if config.model_type == ModelType::ComplEx {
                    dim * 2
                } else {
                    dim
                };
                let rel_cols = ent_cols;
                let ent =
                    Var::rand_f64(-scale, scale, (num_entities, ent_cols), DType::F32, device)?;
                let rel =
                    Var::rand_f64(-scale, scale, (num_relations, rel_cols), DType::F32, device)?;
                (ent, rel)
            }
        };

        Ok(Self {
            entity_embeddings,
            relation_embeddings,
            model_type: config.model_type,
            dim,
            gamma,
            distance_norm: config.distance_norm,
            embedding_dropout: config.embedding_dropout,
            device: device.clone(),
        })
    }

    /// Score a batch of triples. Returns tensor of shape `[batch]`.
    ///
    /// For distance-based models (TransE, RotatE): returns distances (lower = more likely).
    /// For similarity-based models (ComplEx, DistMult): returns negative similarities.
    pub fn score_batch(
        &self,
        heads: &Tensor,
        relations: &Tensor,
        tails: &Tensor,
    ) -> Result<Tensor> {
        let mut h = self.entity_embeddings.as_tensor().index_select(heads, 0)?;
        let mut r = self
            .relation_embeddings
            .as_tensor()
            .index_select(relations, 0)?;
        let mut t = self.entity_embeddings.as_tensor().index_select(tails, 0)?;

        if self.embedding_dropout > 0.0 {
            h = candle_nn::ops::dropout(&h, self.embedding_dropout)?;
            r = candle_nn::ops::dropout(&r, self.embedding_dropout)?;
            t = candle_nn::ops::dropout(&t, self.embedding_dropout)?;
        }

        match self.model_type {
            ModelType::TransE => {
                let diff = ((h + r)? - t)?;
                match self.distance_norm {
                    1 => diff.abs()?.sum(D::Minus1),
                    _ => diff.sqr()?.sum(D::Minus1)?.sqrt(),
                }
            }
            ModelType::RotatE => {
                // Split into re/im pairs.
                let dim = self.dim;
                let h_re = h.i((.., ..dim))?;
                let h_im = h.i((.., dim..))?;
                let t_re = t.i((.., ..dim))?;
                let t_im = t.i((.., dim..))?;
                // r is angles: compute cos/sin.
                let r_cos = r.cos()?;
                let r_sin = r.sin()?;
                // h * r (complex multiply)
                let hr_re = ((&h_re * &r_cos)? - (&h_im * &r_sin)?)?;
                let hr_im = ((&h_re * &r_sin)? + (&h_im * &r_cos)?)?;
                let d_re = (hr_re - t_re)?;
                let d_im = (hr_im - t_im)?;
                match self.distance_norm {
                    1 => {
                        let dist = (d_re.abs()? + d_im.abs()?)?;
                        dist.sum(D::Minus1)
                    }
                    _ => {
                        let dist_sq = (d_re.sqr()? + d_im.sqr()?)?;
                        dist_sq.sum(D::Minus1)?.sqrt()
                    }
                }
            }
            ModelType::ComplEx => {
                let dim = self.dim;
                let h_re = h.i((.., ..dim))?;
                let h_im = h.i((.., dim..))?;
                let r_re = r.i((.., ..dim))?;
                let r_im = r.i((.., dim..))?;
                let t_re = t.i((.., ..dim))?;
                let t_im = t.i((.., dim..))?;
                // h * r (complex)
                let hr_re = ((&h_re * &r_re)? - (&h_im * &r_im)?)?;
                let hr_im = ((&h_re * &r_im)? + (&h_im * &r_re)?)?;
                // Re(hr * conj(t)) = hr_re * t_re + hr_im * t_im
                let score = ((&hr_re * &t_re)? + (&hr_im * &t_im)?)?;
                // Negate so lower = more likely (distance convention).
                score.sum(D::Minus1)?.neg()
            }
            ModelType::DistMult => {
                let score = ((&h * &r)? * &t)?;
                score.sum(D::Minus1)?.neg()
            }
        }
    }

    /// Score all entities as tails for a batch of (h, r) queries.
    ///
    /// Returns tensor of shape `[batch, num_entities]`.
    /// For dot-product models (DistMult, ComplEx), uses matmul.
    /// For distance models (TransE), uses the squared-distance-via-GEMM trick.
    pub fn score_1n(&self, heads: &Tensor, relations: &Tensor) -> Result<Tensor> {
        let h = self.entity_embeddings.as_tensor().index_select(heads, 0)?;
        let r = self
            .relation_embeddings
            .as_tensor()
            .index_select(relations, 0)?;
        let ent_matrix = self.entity_embeddings.as_tensor(); // [E, dim]

        match self.model_type {
            ModelType::TransE => {
                // -||h+r-t||^2 = -(||h+r||^2 - 2*(h+r)@E^T + ||E||^2)
                // We want lower = more likely, so return positive distance.
                // But for BCE, we need higher = more likely. Return negative distance.
                let hr = (h + r)?; // [B, dim]
                let hr_sq = hr.sqr()?.sum(D::Minus1)?; // [B]
                let ent_sq = ent_matrix.sqr()?.sum(D::Minus1)?; // [E]
                let cross = hr.matmul(&ent_matrix.t()?)?; // [B, E]
                                                          // dist^2 = hr_sq - 2*cross + ent_sq
                let dist_sq = (hr_sq
                    .unsqueeze(D::Minus1)?
                    .broadcast_add(&ent_sq.unsqueeze(0)?)?
                    - (cross * 2.0)?)?;
                // Return negative distance (higher = more likely for BCE)
                dist_sq.neg()
            }
            ModelType::DistMult => {
                // score = sum(h * r * t) = (h*r) @ E^T
                let hr = (h * r)?; // [B, dim]
                hr.matmul(&ent_matrix.t()?) // [B, E], higher = more likely
            }
            ModelType::ComplEx => {
                let dim = self.dim;
                let h_re = h.i((.., ..dim))?;
                let h_im = h.i((.., dim..))?;
                let r_re = r.i((.., ..dim))?;
                let r_im = r.i((.., dim..))?;
                let hr_re = ((&h_re * &r_re)? - (&h_im * &r_im)?)?;
                let hr_im = ((&h_re * &r_im)? + (&h_im * &r_re)?)?;
                let e_re = ent_matrix.i((.., ..dim))?;
                let e_im = ent_matrix.i((.., dim..))?;
                // Re(hr * conj(e)) = hr_re @ e_re^T + hr_im @ e_im^T
                let score = (hr_re.matmul(&e_re.t()?)? + hr_im.matmul(&e_im.t()?)?)?;
                Ok(score) // higher = more likely
            }
            ModelType::RotatE => {
                // RotatE isn't a dot product, so 1-N via GEMM isn't straightforward.
                // Fall back to per-entity scoring.
                // TODO: implement distance-via-GEMM for complex rotation
                let dim = self.dim;
                let h_re = h.i((.., ..dim))?;
                let h_im = h.i((.., dim..))?;
                let r_cos = r.cos()?;
                let r_sin = r.sin()?;
                let hr_re = ((&h_re * &r_cos)? - (&h_im * &r_sin)?)?;
                let hr_im = ((&h_re * &r_sin)? + (&h_im * &r_cos)?)?;
                // Concatenate [hr_re, hr_im] -> [B, 2*dim]
                let hr = Tensor::cat(&[&hr_re, &hr_im], D::Minus1)?;
                // Same GEMM trick as TransE but on concatenated complex vectors
                let hr_sq = hr.sqr()?.sum(D::Minus1)?;
                let ent_sq = ent_matrix.sqr()?.sum(D::Minus1)?;
                let cross = hr.matmul(&ent_matrix.t()?)?;
                let dist_sq = (hr_sq
                    .unsqueeze(D::Minus1)?
                    .broadcast_add(&ent_sq.unsqueeze(0)?)?
                    - (cross * 2.0)?)?;
                dist_sq.neg()
            }
        }
    }

    /// Compute N3 regularization: `||h||_3^3 + ||r||_3^3 + ||t||_3^3`.
    fn n3_penalty(&self, heads: &Tensor, relations: &Tensor, tails: &Tensor) -> Result<Tensor> {
        let h = self.entity_embeddings.as_tensor().index_select(heads, 0)?;
        let r = self
            .relation_embeddings
            .as_tensor()
            .index_select(relations, 0)?;
        let t = self.entity_embeddings.as_tensor().index_select(tails, 0)?;
        let cube_norm = |x: &Tensor| -> Result<Tensor> { x.abs()?.powf(3.0)?.mean_all() };
        let penalty = (cube_norm(&h)? + cube_norm(&r)? + cube_norm(&t)?)?;
        Ok(penalty)
    }

    /// Extract entity embeddings as `Vec<Vec<f32>>`.
    pub fn entity_vecs(&self) -> Result<Vec<Vec<f32>>> {
        tensor_to_vecs(self.entity_embeddings.as_tensor())
    }

    /// Extract relation embeddings as `Vec<Vec<f32>>`.
    pub fn relation_vecs(&self) -> Result<Vec<Vec<f32>>> {
        tensor_to_vecs(self.relation_embeddings.as_tensor())
    }

    /// Convert to a CPU-based TransE model for evaluation.
    pub fn to_transe(&self) -> Result<crate::TransE> {
        Ok(crate::TransE::from_vecs(
            self.entity_vecs()?,
            self.relation_vecs()?,
            self.dim,
        ))
    }

    /// Convert to a CPU-based RotatE model for evaluation.
    pub fn to_rotate(&self) -> Result<crate::RotatE> {
        Ok(crate::RotatE::from_vecs(
            self.entity_vecs()?,
            self.relation_vecs()?,
            self.dim,
            self.gamma,
        ))
    }

    /// Convert to a CPU-based ComplEx model for evaluation.
    pub fn to_complex(&self) -> Result<crate::ComplEx> {
        Ok(crate::ComplEx::from_vecs(
            self.entity_vecs()?,
            self.relation_vecs()?,
            self.dim,
        ))
    }

    /// Convert to a CPU-based DistMult model for evaluation.
    pub fn to_distmult(&self) -> Result<crate::DistMult> {
        Ok(crate::DistMult::from_vecs(
            self.entity_vecs()?,
            self.relation_vecs()?,
            self.dim,
        ))
    }
}

/// Training outcome.
pub struct TrainResult {
    /// The trained model.
    pub model: TrainableModel,
    /// Loss per epoch (averaged over batches).
    pub losses: Vec<f32>,
}

/// Validation data for early stopping.
pub struct ValidationData<'a> {
    /// Validation triples to evaluate.
    pub valid_triples: &'a [(usize, usize, usize)],
    /// All known triples (train + valid + test) for filtered evaluation.
    pub all_triples: &'a [(usize, usize, usize)],
}

/// Train a KGE model on the given triples.
///
/// `train_triples` is a slice of `(head, relation, tail)` ID triples.
/// `num_entities` and `num_relations` define the vocabulary size.
///
/// If `validation` is provided and `config.eval_interval > 0`, evaluates
/// on the validation set periodically and stops early if MRR doesn't
/// improve for `config.patience` evaluation cycles.
///
/// Returns the trained model and per-epoch loss history.
pub fn train(
    train_triples: &[(usize, usize, usize)],
    num_entities: usize,
    num_relations: usize,
    config: &TrainConfig,
    device: &Device,
) -> Result<TrainResult> {
    train_with_validation(
        train_triples,
        num_entities,
        num_relations,
        config,
        device,
        None,
    )
}

/// Train with optional validation-based early stopping.
pub fn train_with_validation(
    train_triples: &[(usize, usize, usize)],
    num_entities: usize,
    num_relations: usize,
    config: &TrainConfig,
    device: &Device,
    validation: Option<ValidationData<'_>>,
) -> Result<TrainResult> {
    let model = TrainableModel::new(num_entities, num_relations, config, device)?;
    let vars = vec![
        model.entity_embeddings.clone(),
        model.relation_embeddings.clone(),
    ];
    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: config.lr,
            weight_decay: 0.0,
            ..ParamsAdamW::default()
        },
    )?;

    let n_triples = train_triples.len();
    let batch_size = config.batch_size.min(n_triples);
    let gamma = config.gamma;
    let alpha = config.adversarial_temperature;
    let n3_coeff = config.n3_reg;

    // Precompute entity frequency for optional subsampling weights.
    let entity_freq = if config.subsampling {
        let mut freq = vec![0u32; num_entities];
        for &(h, _, t) in train_triples {
            freq[h] += 1;
            freq[t] += 1;
        }
        Some(freq)
    } else {
        None
    };

    let mut losses = Vec::with_capacity(config.epochs);
    let mut shuffled: Vec<(usize, usize, usize)> = train_triples.to_vec();
    let mut best_mrr = f32::NEG_INFINITY;
    let mut patience_counter = 0_usize;

    let base_lr = config.lr;

    for _epoch in 0..config.epochs {
        // Linear warmup.
        if config.warmup_epochs > 0 && _epoch < config.warmup_epochs {
            let lr = base_lr * (_epoch + 1) as f64 / config.warmup_epochs as f64;
            optimizer.set_learning_rate(lr);
        } else if config.warmup_epochs > 0 && _epoch == config.warmup_epochs {
            optimizer.set_learning_rate(base_lr);
        }

        let mut epoch_loss = 0.0_f64;
        let mut n_batches = 0u32;

        // Shuffle triples each epoch.
        {
            use rand::seq::SliceRandom;
            shuffled.shuffle(&mut rand::rng());
        }

        let mut offset = 0;
        while offset < n_triples {
            let end = (offset + batch_size).min(n_triples);
            let batch = &shuffled[offset..end];
            let actual_bs = batch.len();
            offset = end;

            let heads_data: Vec<u32> = batch.iter().map(|&(h, _, _)| h as u32).collect();
            let rels_data: Vec<u32> = batch.iter().map(|&(_, r, _)| r as u32).collect();
            let tails_data: Vec<u32> = batch.iter().map(|&(_, _, t)| t as u32).collect();

            let heads = Tensor::from_vec(heads_data, actual_bs, &model.device)?;
            let rels = Tensor::from_vec(rels_data, actual_bs, &model.device)?;
            let tails = Tensor::from_vec(tails_data.clone(), actual_bs, &model.device)?;

            let mut loss = if config.one_to_n {
                // 1-N scoring: score all entities as tails, BCE loss.
                let scores = model.score_1n(&heads, &rels)?; // [B, E]

                // Build target: 1-eps for correct tail, eps/E for others.
                let eps = config.label_smoothing as f64;
                let mut target_data =
                    vec![eps as f32 / num_entities as f32; actual_bs * num_entities];
                for (i, &(_, _, t)) in batch.iter().enumerate() {
                    target_data[i * num_entities + t] = 1.0 - eps as f32;
                }
                let targets =
                    Tensor::from_vec(target_data, (actual_bs, num_entities), &model.device)?;

                // BCE loss: -[y*log(sigmoid(s)) + (1-y)*log(sigmoid(-s))]
                let log_sig = log_sigmoid(&scores)?;
                let log_sig_neg = log_sigmoid(&scores.neg()?)?;
                let bce = ((&targets * &log_sig)? + ((1.0 - &targets)? * &log_sig_neg)?)?;
                bce.neg()?.mean_all()?
            } else {
                // Negative sampling with SANS.
                let pos_scores = model.score_batch(&heads, &rels, &tails)?;

                let neg_entities = Tensor::rand(
                    0.0_f32,
                    num_entities as f32,
                    (actual_bs, config.num_negatives),
                    &model.device,
                )?
                .to_dtype(DType::U32)?;

                let corrupt_mask = Tensor::rand(
                    0.0_f32,
                    1.0_f32,
                    (actual_bs, config.num_negatives),
                    &model.device,
                )?;
                let half = Tensor::full(0.5_f32, (actual_bs, config.num_negatives), &model.device)?;
                let corrupt_head = corrupt_mask.lt(&half)?;

                let heads_exp = heads
                    .unsqueeze(1)?
                    .expand((actual_bs, config.num_negatives))?;
                let rels_exp = rels
                    .unsqueeze(1)?
                    .expand((actual_bs, config.num_negatives))?;
                let tails_exp = tails
                    .unsqueeze(1)?
                    .expand((actual_bs, config.num_negatives))?;

                let neg_heads = corrupt_head.where_cond(&neg_entities, &heads_exp)?;
                let neg_tails = corrupt_head.where_cond(&tails_exp, &neg_entities)?;

                let neg_scores = model
                    .score_batch(
                        &neg_heads.flatten_all()?,
                        &rels_exp.flatten_all()?,
                        &neg_tails.flatten_all()?,
                    )?
                    .reshape((actual_bs, config.num_negatives))?;

                let neg_weights = if alpha > 0.0 {
                    let scaled = (neg_scores.detach() * (-(alpha as f64)))?.detach();
                    candle_nn::ops::softmax(&scaled, D::Minus1)?
                } else {
                    Tensor::ones((actual_bs, config.num_negatives), DType::F32, &model.device)?
                        .affine(1.0 / config.num_negatives as f64, 0.0)?
                };

                let pos_loss = log_sigmoid(&(pos_scores.neg()? + gamma as f64)?)?.neg()?;
                let neg_loss_per = log_sigmoid(&(neg_scores - gamma as f64)?)?;
                let weighted_neg_loss = (&neg_weights * &neg_loss_per)?.sum(D::Minus1)?.neg()?;

                let per_triple_loss = (pos_loss + weighted_neg_loss)?;
                if let Some(ref freq) = entity_freq {
                    let subsample_w: Vec<f32> = batch
                        .iter()
                        .map(|&(h, _, t)| 1.0 / ((freq[h] + freq[t]) as f32).sqrt())
                        .collect();
                    let subsample_t = Tensor::from_vec(subsample_w, actual_bs, &model.device)?;
                    (&per_triple_loss * &subsample_t)?.mean_all()?
                } else {
                    per_triple_loss.mean_all()?
                }
            };

            // N3 regularization.
            if n3_coeff > 0.0 {
                let n3 = model.n3_penalty(&heads, &rels, &tails)?;
                loss = (loss + (n3 * n3_coeff as f64)?)?;
            }

            optimizer.backward_step(&loss)?;

            // Entity normalization (L2 unit norm per row).
            if config.normalize_entities {
                let ent = model.entity_embeddings.as_tensor();
                let norms = ent.sqr()?.sum(D::Minus1)?.sqrt()?.unsqueeze(D::Minus1)?;
                let normalized = ent.broadcast_div(&norms.clamp(1e-8, f64::MAX)?)?;
                model.entity_embeddings.set(&normalized)?;
            }

            epoch_loss += loss.to_scalar::<f32>()? as f64;
            n_batches += 1;
        }

        let avg_loss = (epoch_loss / n_batches as f64) as f32;
        losses.push(avg_loss);

        if config.log_interval > 0 && (_epoch + 1) % config.log_interval == 0 {
            eprintln!("epoch {:>4} | loss {:.4}", _epoch + 1, avg_loss);
        }

        // Validation-based early stopping.
        if let Some(ref val) = validation {
            if config.eval_interval > 0 && (_epoch + 1) % config.eval_interval == 0 {
                let scorer: Box<dyn crate::Scorer + Sync> = match model.model_type {
                    ModelType::TransE => Box::new(model.to_transe()?),
                    ModelType::RotatE => Box::new(model.to_rotate()?),
                    ModelType::ComplEx => Box::new(model.to_complex()?),
                    ModelType::DistMult => Box::new(model.to_distmult()?),
                };
                let metrics = crate::eval::evaluate_link_prediction(
                    scorer.as_ref(),
                    val.valid_triples,
                    val.all_triples,
                    num_entities,
                );
                if metrics.mrr > best_mrr {
                    best_mrr = metrics.mrr;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= config.patience {
                        eprintln!(
                            "Early stopping at epoch {} (best MRR: {:.4})",
                            _epoch + 1,
                            best_mrr,
                        );
                        break;
                    }
                }
            }
        }
    }

    Ok(TrainResult { model, losses })
}

/// Numerically stable `log(sigmoid(x))`.
fn log_sigmoid(x: &Tensor) -> Result<Tensor> {
    // log(sigmoid(x)) = x - softplus(x) = x - log(1 + exp(x))
    // For numerical stability: -softplus(-x) = -log(1 + exp(-x))
    // Use: log(sigmoid(x)) = -max(0, -x) - log(1 + exp(-|x|))
    let neg_x = x.neg()?;
    let abs_x = x.abs()?;
    let neg_abs = abs_x.neg()?;
    // relu(-x) = max(0, -x)
    let relu_neg = neg_x.relu()?;
    // log(1 + exp(-|x|)) -- no log1p in candle, use log(exp(-|x|) + 1)
    let softplus = (neg_abs.exp()? + 1.0)?.log()?;
    let result = (relu_neg.neg()? - softplus)?;
    Ok(result)
}

fn tensor_to_vecs(t: &Tensor) -> Result<Vec<Vec<f32>>> {
    let t = t.to_device(&Device::Cpu)?;
    let rows = t.dim(0)?;
    let cols = t.dim(1)?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    Ok((0..rows)
        .map(|i| data[i * cols..(i + 1) * cols].to_vec())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sigmoid_basic() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0_f32, 10.0, -10.0], &device).unwrap();
        let result = log_sigmoid(&x).unwrap().to_vec1::<f32>().unwrap();
        // log(sigmoid(0)) = log(0.5) ~ -0.693
        assert!((result[0] - (-0.693)).abs() < 0.01, "got {}", result[0]);
        // log(sigmoid(10)) ~ 0
        assert!(result[1] > -0.001, "got {}", result[1]);
        // log(sigmoid(-10)) ~ -10
        assert!((result[2] - (-10.0)).abs() < 0.01, "got {}", result[2]);
    }

    #[test]
    fn train_transe_smoke() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2), (2, 1, 0), (0, 1, 2)];
        let config = TrainConfig {
            model_type: ModelType::TransE,
            dim: 8,
            num_negatives: 4,
            gamma: 6.0,
            adversarial_temperature: 0.5,
            lr: 0.01,
            n3_reg: 0.0,
            batch_size: 4,
            epochs: 5,
            ..TrainConfig::default()
        };
        let result = train(&triples, 3, 2, &config, &device).unwrap();
        assert_eq!(result.losses.len(), 5);
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let model = result.model.to_transe().unwrap();
        assert_eq!(model.entities().len(), 3);
    }

    #[test]
    fn train_rotate_smoke() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2)];
        let config = TrainConfig {
            model_type: ModelType::RotatE,
            dim: 4,
            num_negatives: 2,
            gamma: 6.0,
            adversarial_temperature: 1.0,
            lr: 0.01,
            n3_reg: 0.0,
            batch_size: 2,
            epochs: 3,
            ..TrainConfig::default()
        };
        let result = train(&triples, 3, 1, &config, &device).unwrap();
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let model = result.model.to_rotate().unwrap();
        assert_eq!(model.entities().len(), 3);
    }

    #[test]
    fn train_complex_with_n3() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2)];
        let config = TrainConfig {
            model_type: ModelType::ComplEx,
            dim: 4,
            num_negatives: 2,
            gamma: 6.0,
            adversarial_temperature: 1.0,
            lr: 0.01,
            n3_reg: 0.001,
            batch_size: 2,
            epochs: 3,
            ..TrainConfig::default()
        };
        let result = train(&triples, 3, 1, &config, &device).unwrap();
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let model = result.model.to_complex().unwrap();
        assert_eq!(model.entities().len(), 3);
    }

    #[test]
    fn train_distmult_smoke() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2)];
        let config = TrainConfig {
            model_type: ModelType::DistMult,
            dim: 8,
            num_negatives: 2,
            gamma: 6.0,
            adversarial_temperature: 0.0,
            lr: 0.01,
            n3_reg: 0.0,
            batch_size: 2,
            epochs: 3,
            ..TrainConfig::default()
        };
        let result = train(&triples, 3, 1, &config, &device).unwrap();
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let model = result.model.to_distmult().unwrap();
        assert_eq!(model.entities().len(), 3);
    }

    #[test]
    fn loss_decreases() {
        let device = Device::Cpu;
        // Enough data and epochs for loss to decrease.
        let triples: Vec<_> = (0..20).map(|i| (i % 10, i % 3, (i + 1) % 10)).collect();
        let config = TrainConfig {
            model_type: ModelType::TransE,
            dim: 16,
            num_negatives: 8,
            gamma: 6.0,
            adversarial_temperature: 0.5,
            lr: 0.01,
            n3_reg: 0.0,
            batch_size: 10,
            epochs: 50,
            ..TrainConfig::default()
        };
        let result = train(&triples, 10, 3, &config, &device).unwrap();
        let first = result.losses[0];
        let last = *result.losses.last().unwrap();
        assert!(
            last < first,
            "Loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn transe_achieves_nonzero_mrr_on_trivial_graph() {
        // 5 entities, 1 relation: 0->1, 1->2, 2->3, 3->4.
        // After training, score(0,0,1) should be the best among all tails.
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 4)];
        let config = TrainConfig {
            model_type: ModelType::TransE,
            dim: 32,
            num_negatives: 4,
            gamma: 6.0,
            adversarial_temperature: 0.0,
            lr: 0.01,
            n3_reg: 0.0,
            batch_size: 4,
            epochs: 500,
            ..TrainConfig::default()
        };
        let result = train(&triples, 5, 1, &config, &device).unwrap();
        let model = result.model.to_transe().unwrap();

        let metrics = crate::eval::evaluate_link_prediction(&model, &triples, &triples, 5);
        assert!(
            metrics.mrr > 0.3,
            "TransE should achieve MRR > 0.3 on trivial graph, got {:.4}",
            metrics.mrr
        );
    }

    #[test]
    fn one_to_n_distmult_smoke() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2), (2, 1, 0), (0, 1, 2)];
        let config = TrainConfig {
            model_type: ModelType::DistMult,
            dim: 8,
            one_to_n: true,
            label_smoothing: 0.1,
            lr: 0.01,
            batch_size: 4,
            epochs: 10,
            ..TrainConfig::default()
        };
        let result = train(&triples, 3, 2, &config, &device).unwrap();
        assert_eq!(result.losses.len(), 10);
        assert!(result.losses.iter().all(|l| l.is_finite()));
        // Loss should decrease with 1-N.
        let first = result.losses[0];
        let last = *result.losses.last().unwrap();
        assert!(last < first, "1-N loss should decrease: {first} -> {last}");
    }

    #[test]
    fn one_to_n_transe_smoke() {
        let device = Device::Cpu;
        let triples = vec![(0, 0, 1), (1, 0, 2), (2, 0, 3)];
        let config = TrainConfig {
            model_type: ModelType::TransE,
            dim: 8,
            one_to_n: true,
            label_smoothing: 0.1,
            lr: 0.001,
            batch_size: 3,
            epochs: 10,
            ..TrainConfig::default()
        };
        let result = train(&triples, 4, 1, &config, &device).unwrap();
        assert!(result.losses.iter().all(|l| l.is_finite()));
    }
}
