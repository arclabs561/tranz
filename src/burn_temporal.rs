//! Burn trainer for [`crate::temporal::TComplEx`].
//!
//! Same recipe as the static [`train_kge`](crate::burn_train::train_kge)
//! trainer (1-N cross-entropy over all entities, both directions, AdamW,
//! optional label smoothing) plus the temporal pieces: a third embedding
//! table for timestamps, `q = r ∘ w_τ` in place of the relation, and the
//! paper's two regularizers (weighted nuclear-3 Ω³ and the Λ₃ temporal
//! smoothness prior; Lacroix et al., ICLR 2020, Eqs. 4-5).

use burn::module::{Module, Param, ParamId};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::{activation, Int, Tensor};

use crate::burn_train::BurnTrainConfig;
use crate::temporal::{QuadIds, TComplEx};

/// TComplEx model: complex entity, relation, and timestamp tables.
#[derive(Module, Debug)]
pub struct BurnTComplEx<B: Backend> {
    entity: Param<Tensor<B, 2>>,
    relation: Param<Tensor<B, 2>>,
    time: Param<Tensor<B, 2>>,
}

/// Result of TComplEx training.
pub struct TComplExResult {
    /// Entity embeddings (`2 * dim` per row).
    pub entity_vecs: Vec<Vec<f32>>,
    /// Relation embeddings (`2 * dim` per row).
    pub relation_vecs: Vec<Vec<f32>>,
    /// Timestamp embeddings (`2 * dim` per row, chronological row order).
    pub time_vecs: Vec<Vec<f32>>,
    /// Complex dimension.
    pub dim: usize,
    /// Loss per epoch.
    pub losses: Vec<f32>,
}

impl TComplExResult {
    /// Build the matching CPU scorer.
    pub fn to_scorer(&self) -> TComplEx {
        TComplEx::from_vecs(
            self.entity_vecs.clone(),
            self.relation_vecs.clone(),
            self.time_vecs.clone(),
            self.dim,
        )
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

/// Componentwise complex product of `[B, 2*dim]` tensors.
fn complex_mul<B: Backend>(
    a: Tensor<B, 2>,
    b: Tensor<B, 2>,
    dim: usize,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let (a_re, a_im) = re_im(a, dim);
    let (b_re, b_im) = re_im(b, dim);
    (
        a_re.clone() * b_re.clone() - a_im.clone() * b_im.clone(),
        a_re * b_im + a_im * b_re,
    )
}

/// Score all entities as tails for `(h, r, ?, τ)`: `Re(<h, r∘w, conj(e)>)`
/// for every entity `e`. Higher = more likely. `[B, E]`.
fn score_1n_tails<B: Backend>(
    model: &BurnTComplEx<B>,
    dim: usize,
    heads: &Tensor<B, 1, Int>,
    rels: &Tensor<B, 1, Int>,
    times: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let h = model.entity.val().select(0, heads.clone());
    let r = model.relation.val().select(0, rels.clone());
    let w = model.time.val().select(0, times.clone());
    let (q_re, q_im) = complex_mul(r, w, dim);
    let (h_re, h_im) = re_im(h, dim);
    let c_re = h_re.clone() * q_re.clone() - h_im.clone() * q_im.clone();
    let c_im = h_re * q_im + h_im * q_re;
    let (e_re, e_im) = re_im(model.entity.val(), dim);
    c_re.matmul(e_re.transpose()) + c_im.matmul(e_im.transpose())
}

/// Score all entities as heads for `(?, r, t, τ)`. With `c = (r∘w) ∘ conj(t)`,
/// `Re(<e, r∘w, conj(t)>) = e_re · c_re − e_im · c_im`. Higher = more likely.
fn score_1n_heads<B: Backend>(
    model: &BurnTComplEx<B>,
    dim: usize,
    rels: &Tensor<B, 1, Int>,
    tails: &Tensor<B, 1, Int>,
    times: &Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let r = model.relation.val().select(0, rels.clone());
    let w = model.time.val().select(0, times.clone());
    let t = model.entity.val().select(0, tails.clone());
    let (q_re, q_im) = complex_mul(r, w, dim);
    let (t_re, t_im) = re_im(t, dim);
    // q ∘ conj(t).
    let c_re = q_re.clone() * t_re.clone() + q_im.clone() * t_im.clone();
    let c_im = q_im * t_re - q_re * t_im;
    let (e_re, e_im) = re_im(model.entity.val(), dim);
    c_re.matmul(e_re.transpose()) - c_im.matmul(e_im.transpose())
}

/// Train TComplEx with 1-N (1vsAll) scoring in both directions.
///
/// Two regularizers from Lacroix et al. (ICLR 2020), both off at `0`:
///
/// - `config.n3_reg`: the weighted nuclear-3 variational form Ω³ (their
///   Eq. 4): per sampled quad, `(λ/3)(‖h‖₃³ + ‖t‖₃³ + ‖r ∘ w‖₃³)` with the
///   complex modulus per component. The (relation, timestamp) pair is
///   penalized JOINTLY as one factor: unfolding the order-4 tensor along
///   the predicate and time modes makes `r ∘ w` a single mode-row, which
///   weights by the joint (predicate, timestamp) marginal rather than the
///   product of marginals (their Appendix 8.1/8.3).
/// - `time_smooth`: the temporal smoothness prior Λ_p (their Eq. 5) with
///   `p = 3` to match Ω³'s order: `λ_t/(|T|−1) · Σ |t_{i+1} − t_i|³`.
///   Timestamp ids must be in chronological order (the loader guarantees
///   this) or the penalty ties the wrong neighbors.
pub fn train_tcomplex<B: AutodiffBackend>(
    train_quads: &[QuadIds],
    num_entities: usize,
    num_relations: usize,
    num_timestamps: usize,
    config: &BurnTrainConfig,
    time_smooth: f64,
    device: &B::Device,
) -> TComplExResult {
    let cols = 2 * config.dim;
    let mk = |rows: usize| {
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
    let mut model = BurnTComplEx {
        entity: mk(num_entities),
        relation: mk(num_relations),
        time: mk(num_timestamps),
    };
    let mut optim = AdamWConfig::new()
        .with_epsilon(1e-8)
        .with_weight_decay(0.0)
        .init::<B, BurnTComplEx<B>>();

    let n = train_quads.len();
    let batch_size = config.batch_size.min(n).max(1);
    let eps = config.label_smoothing;
    let mut losses = Vec::with_capacity(config.epochs);
    let mut indices: Vec<usize> = (0..n).collect();

    for _epoch in 0..config.epochs {
        {
            use rand::seq::SliceRandom;
            indices.shuffle(&mut rand::rng());
        }
        // Detached on-device loss accumulator; one sync per epoch (see
        // train_kge for why per-batch into_scalar stalls wgpu/Metal).
        let mut epoch_loss_acc: Option<Tensor<<B as AutodiffBackend>::InnerBackend, 1>> = None;
        let mut n_batches = 0u32;
        let mut offset = 0;
        while offset < n {
            let end = (offset + batch_size).min(n);
            let batch_idx = &indices[offset..end];
            let bs = batch_idx.len();
            offset = end;

            let gather = |f: fn(&QuadIds) -> usize| -> Tensor<B, 1, Int> {
                let v: Vec<i64> = batch_idx
                    .iter()
                    .map(|&i| f(&train_quads[i]) as i64)
                    .collect();
                Tensor::from_data(burn::tensor::TensorData::new(v, [bs]), device)
            };
            let heads = gather(|q| q.head);
            let rels = gather(|q| q.relation);
            let tails = gather(|q| q.tail);
            let times = gather(|q| q.time);

            let current = model.clone();
            let tail_lp = activation::log_softmax(
                score_1n_tails(&current, config.dim, &heads, &rels, &times),
                1,
            );
            let head_lp = activation::log_softmax(
                score_1n_heads(&current, config.dim, &rels, &tails, &times),
                1,
            );
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
            if config.n3_reg > 0.0 {
                // Ω³ on the sampled rows; q = r ∘ w is one joint factor.
                let h = current.entity.val().select(0, heads.clone());
                let t = current.entity.val().select(0, tails.clone());
                let r = current.relation.val().select(0, rels.clone());
                let w = current.time.val().select(0, times.clone());
                let (q_re, q_im) = complex_mul(r, w, config.dim);
                let mod3 = |re: Tensor<B, 2>, im: Tensor<B, 2>| {
                    (re.powf_scalar(2.0) + im.powf_scalar(2.0))
                        .add_scalar(1e-12)
                        .sqrt()
                        .powf_scalar(3.0)
                        .sum_dim(1)
                };
                let (h_re, h_im) = re_im(h, config.dim);
                let (t_re, t_im) = re_im(t, config.dim);
                let omega3 = (mod3(h_re, h_im) + mod3(t_re, t_im) + mod3(q_re, q_im)).mean();
                loss = loss + omega3.mul_scalar(config.n3_reg / 3.0);
            }
            if time_smooth > 0.0 && num_timestamps > 1 {
                // Λ₃ on the discrete derivative of the timestamp table.
                let w = current.time.val();
                let later = w.clone().slice([1..num_timestamps, 0..cols]);
                let earlier = w.slice([0..num_timestamps - 1, 0..cols]);
                let smooth = (later - earlier)
                    .abs()
                    .powf_scalar(3.0)
                    .sum()
                    .div_scalar((num_timestamps - 1) as f32);
                loss = loss + smooth.mul_scalar(time_smooth);
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

    let extract = |p: &Param<Tensor<B, 2>>| -> Vec<Vec<f32>> {
        let data: Vec<f32> = p.val().into_data().to_vec().unwrap();
        data.chunks(cols).map(<[f32]>::to_vec).collect()
    };
    TComplExResult {
        entity_vecs: extract(&model.entity),
        relation_vecs: extract(&model.relation),
        time_vecs: extract(&model.time),
        dim: config.dim,
        losses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::{evaluate_temporal_link_prediction, TemporalFilterIndex, TemporalScorer};

    #[cfg(feature = "burn-ndarray")]
    type TestBackend = burn::backend::Autodiff<burn_ndarray::NdArray>;

    #[cfg(feature = "burn-ndarray")]
    fn test_device() -> <TestBackend as Backend>::Device {
        burn_ndarray::NdArrayDevice::Cpu
    }

    fn qid(h: usize, r: usize, t: usize, tau: usize) -> QuadIds {
        QuadIds::new(h, r, t, tau)
    }

    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn tcomplex_smoke_and_loss_decreases() {
        let quads: Vec<_> = (0..24)
            .map(|i| qid(i % 6, i % 2, (i + 1) % 6, i % 3))
            .collect();
        let config = BurnTrainConfig {
            dim: 8,
            epochs: 20,
            batch_size: 12,
            lr: 0.01,
            ..BurnTrainConfig::default()
        };
        let result = train_tcomplex::<TestBackend>(&quads, 6, 2, 3, &config, 0.01, &test_device());
        assert_eq!(result.losses.len(), 20);
        assert!(result.losses.iter().all(|l| l.is_finite()));
        let first = result.losses[0];
        let last = *result.losses.last().unwrap();
        assert!(last < first, "loss should decrease: {first} -> {last}");
        let scorer = result.to_scorer();
        assert_eq!(scorer.num_entities(), 6);
        assert_eq!(scorer.num_timestamps(), 3);
    }

    /// The burn scoring paths and the CPU scorer must agree on identical
    /// weights: energies are negated scores, so burn + cpu ≈ 0. Guards the
    /// head-role conjugation (minus on the imaginary matmul).
    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn burn_scores_match_cpu_reference() {
        type B = burn_ndarray::NdArray;
        let device = burn_ndarray::NdArrayDevice::Cpu;
        let (ne, nr, nt, dim) = (5usize, 2usize, 3usize, 4usize);
        let val = |i: usize| ((i * 37 + 11) % 19) as f32 / 19.0 - 0.5;
        let rows = |n: usize, seed: usize| -> Vec<Vec<f32>> {
            (0..n)
                .map(|k| (0..dim * 2).map(|j| val(seed + k * 31 + j)).collect())
                .collect()
        };
        let (ent, rel, tim) = (rows(ne, 0), rows(nr, 7), rows(nt, 13));
        let flat = |v: &Vec<Vec<f32>>| -> Tensor<B, 2> {
            let data: Vec<f32> = v.iter().flatten().copied().collect();
            Tensor::from_data(
                burn::tensor::TensorData::new(data, [v.len(), dim * 2]),
                &device,
            )
        };
        let model = BurnTComplEx::<B> {
            entity: Param::initialized(ParamId::new(), flat(&ent)),
            relation: Param::initialized(ParamId::new(), flat(&rel)),
            time: Param::initialized(ParamId::new(), flat(&tim)),
        };
        let cpu = TComplEx::from_vecs(ent, rel, tim, dim);

        let idx = |v: Vec<i64>| -> Tensor<B, 1, Int> {
            let n = v.len();
            Tensor::from_data(burn::tensor::TensorData::new(v, [n]), &device)
        };
        let tails = score_1n_tails(
            &model,
            dim,
            &idx(vec![0, 3]),
            &idx(vec![0, 1]),
            &idx(vec![1, 2]),
        );
        let tails: Vec<f32> = tails.into_data().to_vec().unwrap();
        for (b, (h, r, tau)) in [(0usize, 0usize, 1usize), (3, 1, 2)].iter().enumerate() {
            for e in 0..ne {
                let burn_s = tails[b * ne + e];
                let cpu_e = cpu.score(*h, *r, e, *tau);
                assert!(
                    (burn_s + cpu_e).abs() < 1e-4,
                    "tail parity ({h},{r},{e},{tau}): burn {burn_s} cpu {cpu_e}"
                );
            }
        }
        let heads = score_1n_heads(
            &model,
            dim,
            &idx(vec![0, 1]),
            &idx(vec![2, 4]),
            &idx(vec![0, 2]),
        );
        let heads: Vec<f32> = heads.into_data().to_vec().unwrap();
        for (b, (r, t, tau)) in [(0usize, 2usize, 0usize), (1, 4, 2)].iter().enumerate() {
            for e in 0..ne {
                let burn_s = heads[b * ne + e];
                let cpu_e = cpu.score(e, *r, *t, *tau);
                assert!(
                    (burn_s + cpu_e).abs() < 1e-4,
                    "head parity ({e},{r},{t},{tau}): burn {burn_s} cpu {cpu_e}"
                );
            }
        }
    }

    /// Train on a planted temporal pattern and check the model ranks the
    /// held-out quad well: relation 0 means "successor" at τ=0 but
    /// "predecessor" at τ=1, so time embeddings are load-bearing.
    #[test]
    #[cfg(feature = "burn-ndarray")]
    fn learns_time_dependent_relation() {
        let ne = 6;
        let mut quads = Vec::new();
        for h in 0..ne {
            quads.push(qid(h, 0, (h + 1) % ne, 0));
            quads.push(qid(h, 0, (h + ne - 1) % ne, 1));
        }
        // Hold out one quad per timestamp.
        let test = vec![quads.remove(0), quads.pop().unwrap()];
        let config = BurnTrainConfig {
            dim: 16,
            epochs: 200,
            batch_size: 32,
            lr: 0.02,
            init_scale: 1e-2,
            ..BurnTrainConfig::default()
        };
        let result = train_tcomplex::<TestBackend>(&quads, ne, 1, 2, &config, 0.0, &test_device());
        let scorer = result.to_scorer();
        let mut all = quads.clone();
        all.extend_from_slice(&test);
        let filter = TemporalFilterIndex::from_quads(&all);
        let m = evaluate_temporal_link_prediction(&scorer, &test, &filter);
        assert!(
            m.mrr > 0.5,
            "held-out temporal quads should rank near the top: mrr {}",
            m.mrr
        );
    }
}
