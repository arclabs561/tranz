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
    /// Number of negative samples per positive.
    pub num_negatives: usize,
    /// Margin gamma for the loss function.
    pub gamma: f32,
    /// SANS adversarial temperature. 0 = uniform weighting.
    pub adversarial_temperature: f32,
    /// Learning rate.
    pub lr: f64,
    /// N3 regularization coefficient. 0 = disabled.
    pub n3_reg: f32,
    /// Batch size.
    pub batch_size: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Normalize entity embeddings to unit L2 norm after each step.
    /// Standard for TransE (Bordes et al., 2013). Disabled by default.
    pub normalize_entities: bool,
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
            gamma: 12.0,
            adversarial_temperature: 1.0,
            lr: 0.001,
            n3_reg: 0.0,
            batch_size: 512,
            epochs: 1000,
            normalize_entities: false,
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
                let ent = Var::rand_f64(0.0 - scale, scale, (num_entities, dim), DType::F32, device)?;
                let rel = Var::rand_f64(0.0 - scale, scale, (num_relations, dim), DType::F32, device)?;
                (ent, rel)
            }
            ModelType::RotatE => {
                // Entities: interleaved re/im, so dim*2 columns.
                let ent_scale = gamma as f64 / (dim as f64).sqrt();
                let ent = Var::rand_f64(-ent_scale, ent_scale, (num_entities, dim * 2), DType::F32, device)?;
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
                let ent_cols = if config.model_type == ModelType::ComplEx { dim * 2 } else { dim };
                let rel_cols = ent_cols;
                let ent = Var::rand_f64(-scale, scale, (num_entities, ent_cols), DType::F32, device)?;
                let rel = Var::rand_f64(-scale, scale, (num_relations, rel_cols), DType::F32, device)?;
                (ent, rel)
            }
        };

        Ok(Self {
            entity_embeddings,
            relation_embeddings,
            model_type: config.model_type,
            dim,
            gamma,
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
        let h = self.entity_embeddings.as_tensor().index_select(heads, 0)?;
        let r = self.relation_embeddings.as_tensor().index_select(relations, 0)?;
        let t = self.entity_embeddings.as_tensor().index_select(tails, 0)?;

        match self.model_type {
            ModelType::TransE => {
                // ||h + r - t||_2
                let diff = ((h + r)? - t)?;
                diff.sqr()?.sum(D::Minus1)?.sqrt()
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
                let dist_sq = (d_re.sqr()? + d_im.sqr()?)?;
                dist_sq.sum(D::Minus1)?.sqrt()
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

    /// Compute N3 regularization: `||h||_3^3 + ||r||_3^3 + ||t||_3^3`.
    fn n3_penalty(
        &self,
        heads: &Tensor,
        relations: &Tensor,
        tails: &Tensor,
    ) -> Result<Tensor> {
        let h = self.entity_embeddings.as_tensor().index_select(heads, 0)?;
        let r = self.relation_embeddings.as_tensor().index_select(relations, 0)?;
        let t = self.entity_embeddings.as_tensor().index_select(tails, 0)?;
        let cube_norm = |x: &Tensor| -> Result<Tensor> {
            x.abs()?.powf(3.0)?.mean_all()
        };
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
    train_with_validation(train_triples, num_entities, num_relations, config, device, None)
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

    let mut losses = Vec::with_capacity(config.epochs);
    let mut shuffled: Vec<(usize, usize, usize)> = train_triples.to_vec();
    let mut best_mrr = f32::NEG_INFINITY;
    let mut patience_counter = 0_usize;

    for _epoch in 0..config.epochs {
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
            let tails = Tensor::from_vec(tails_data, actual_bs, &model.device)?;

            // Score positives: [batch]
            let pos_scores = model.score_batch(&heads, &rels, &tails)?;

            // Generate random entities for corruption: [batch, k]
            let neg_entities = Tensor::rand(
                0.0_f32,
                num_entities as f32,
                (actual_bs, config.num_negatives),
                &model.device,
            )?
            .to_dtype(DType::U32)?;

            // Corrupt head or tail with 50/50 probability per sample.
            // corrupt_head: [batch, k] mask where 1 = corrupt head, 0 = corrupt tail.
            let corrupt_mask = Tensor::rand(
                0.0_f32,
                1.0_f32,
                (actual_bs, config.num_negatives),
                &model.device,
            )?;
            let half = Tensor::full(0.5_f32, (actual_bs, config.num_negatives), &model.device)?;
            let corrupt_head = corrupt_mask.lt(&half)?; // 1 where < 0.5

            let heads_exp = heads.unsqueeze(1)?.expand((actual_bs, config.num_negatives))?;
            let rels_exp = rels.unsqueeze(1)?.expand((actual_bs, config.num_negatives))?;
            let tails_exp = tails.unsqueeze(1)?.expand((actual_bs, config.num_negatives))?;

            // Where corrupt_head=1: use neg_entities as head, original tail.
            // Where corrupt_head=0: use original head, neg_entities as tail.
            let neg_heads = corrupt_head.where_cond(&neg_entities, &heads_exp)?;
            let neg_tails = corrupt_head.where_cond(&tails_exp, &neg_entities)?;

            let neg_scores = model.score_batch(
                &neg_heads.flatten_all()?,
                &rels_exp.flatten_all()?,
                &neg_tails.flatten_all()?,
            )?
            .reshape((actual_bs, config.num_negatives))?;

            // SANS weighting (detached -- no gradient through weights).
            let neg_weights = if alpha > 0.0 {
                let scaled = (neg_scores.detach() * (-(alpha as f64)))?.detach();
                candle_nn::ops::softmax(&scaled, D::Minus1)?
            } else {
                Tensor::ones(
                    (actual_bs, config.num_negatives),
                    DType::F32,
                    &model.device,
                )?
                .affine(1.0 / config.num_negatives as f64, 0.0)?
            };

            // Loss: -log(sigma(gamma - pos_score)) - sum_i w_i * log(sigma(neg_score_i - gamma))
            let pos_loss = log_sigmoid(&(pos_scores.neg()? + gamma as f64)?)?.neg()?;
            let neg_loss_per = log_sigmoid(&(neg_scores - gamma as f64)?)?;
            let weighted_neg_loss = (&neg_weights * &neg_loss_per)?.sum(D::Minus1)?.neg()?;

            let mut loss = (pos_loss + weighted_neg_loss)?.mean_all()?;

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

        losses.push((epoch_loss / n_batches as f64) as f32);

        // Validation-based early stopping.
        if let Some(ref val) = validation {
            if config.eval_interval > 0 && (_epoch + 1) % config.eval_interval == 0 {
                let scorer: Box<dyn crate::Scorer> = match model.model_type {
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
        let triples = vec![
            (0, 0, 1),
            (1, 0, 2),
            (2, 1, 0),
            (0, 1, 2),
        ];
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
        let triples: Vec<_> = (0..20)
            .map(|i| (i % 10, i % 3, (i + 1) % 10))
            .collect();
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
}
