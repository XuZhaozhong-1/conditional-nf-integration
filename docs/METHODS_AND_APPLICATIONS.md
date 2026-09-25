# Conditional Normalizing-Flow Integration

## Purpose

This project learns one importance-sampling proposal for a continuously
parameterized family of integrals

\[
I(c)=\int_{[0,1]^d} f(x\mid c)\,dx,
\]

where (x) denotes integration coordinates and (c) is an external
condition. In the scattering applications, $c=\sqrt{\hat s}$, the partonic
center-of-mass energy. The model is trained over an interval of conditions and
then evaluated at held-out conditions without retraining.

The method does not replace Monte Carlo integration with a neural-network
prediction of (I(c)). The network supplies a proposal density, while exact
importance weights retain the original integrand in the estimator.

## Marginal--copula factorization

For a (d)-dimensional proposal, the implementation separates one-dimensional
behavior from dependence:

\[
q_\theta(x\mid c)
=c_\theta(u_1,\ldots,u_d\mid c)
 \prod_{i=1}^d q_{i,\theta}(x_i\mid c),
\qquad
u_i=F_{i,\theta}(x_i\mid c).
\]

Here $q_{i,\theta}$ and $F_{i,\theta}$ are the conditional marginal density
and CDF for coordinate $i$. The copula density $c_\theta$ models the
remaining dependence after each coordinate has been transformed to its
conditional rank $u_i\in[0,1]$.

This decomposition is useful because narrow or asymmetric behavior in a single
coordinate can be learned by its marginal, while curved, multimodal, or
higher-dimensional correlations are handled separately by the copula.

### Conditional marginal maps

Each marginal is a monotone cubic-Hermite map from the unit interval to itself.
A small neural network takes the condition features as input and predicts
positive bin widths and heights. PCHIP-style knot derivatives preserve
monotonicity. The inverse uses bracketed per-bin bisection followed by one
Newton correction, safeguarded on the physical interval $[0,1]$.

For the process sweep, the scalar normalized condition

\[
z=\frac{\sqrt{\hat s}-\sqrt{\hat s}_{\min}}
        {\sqrt{\hat s}_{\max}-\sqrt{\hat s}_{\min}}
\]

is represented by the three features

\[
(z,\ z^2,\ \log(1+z)/\log 2).
\]

### Autoregressive copula

After marginal transformation, an autoregressive copula models each (u_i)
conditioned on the external condition and selected preceding coordinates. A
directed parent graph specifies these dependencies. The reported scattering
process runs use the full autoregressive graph; the repository also contains
experiments for graph selection and pruning, but those are not required for
the process-sweep results.

## Defensive importance sampling

The learned density is mixed with a uniform proposal,

\[
r_\theta(x\mid c)
=(1-\varepsilon)q_\theta(x\mid c)+\varepsilon,
\]

on the unit hypercube. This defensive component gives the proposal support
throughout the integration domain and reduces the risk of missing regions that
the current learned model assigns very little probability.

Fresh samples $x_n\sim r_\theta(\cdot\mid c)$ produce the estimator

\[
\widehat I_N(c)
=\frac1N\sum_{n=1}^N
\frac{f(x_n\mid c)}{r_\theta(x_n\mid c)}.
\]

Consequently, an imperfect neural proposal affects variance rather than
silently replacing the physical integrand. The principal proposal diagnostic
is normalized effective sample size,

\[
\frac{\mathrm{ESS}}{N}
=\frac{(\sum_n w_n)^2}{N\sum_n w_n^2},
\qquad
w_n=\frac{f(x_n\mid c)}{r_\theta(x_n\mid c)}.
\]

Values near one indicate nearly constant importance weights at that condition.

## Training from an unnormalized integrand

Training receives evaluations of the unnormalized nonnegative integrand
$f(x\mid c)$; it does not require exact target samples or the value of
$I(c)$. The workflow is:

1. sample conditions across the training interval;
2. construct self-normalized importance-weighted approximations to the target;
3. learn conditional marginal maps;
4. transform samples to marginal rank coordinates and learn the copula;
5. refresh the weighted cache with the improved proposal;
6. refine at the full target with importance-weighted KL training; and
7. optionally fine-tune toward lower weight variance with a chi-squared
   objective, retaining the update only when held-out ESS improves.

Annealed intermediate targets make the transition from broad initial sampling
to the final sharp integrand more stable. Validation uses fresh samples rather
than reporting the samples used to optimize the model.

## Application to scattering phase space

MadGraph supplies generated matrix elements. A process-specific phase-space
map converts $x\in[0,1]^d$ to on-shell final-state four-momenta and returns
the associated Jacobian. At fixed partonic energy, the implemented integrand
has the schematic form

\[
f(x\mid\sqrt{\hat s})
=\frac{|\mathcal M(p(x))|^2}{2\hat s}\,J(x).
\]

The study is currently parton-level and leading order. The generic process
campaign uses a fixed value of $\alpha_s$ and does not include hadronic PDFs.
The 20-dimensional fully decayed $gg\to t\bar tH$ benchmark uses a dedicated
resonance-aware phase-space map.

The same conditional strategy has been tested at several levels of process
complexity:

| Process | Dimension | Role | Held-out ESS/N |
|---|---:|---|---:|
| $gg\to t\bar t$ | 2 | massive two-body control | 0.981--0.998 |
| $gg\to t\bar tH$ | 5 | massive three-body phase space | 0.959--0.987 |
| $u\bar u\to W^+W^-\to e^+\nu_e\mu^-\bar\nu_\mu$ | 8 | doubly resonant four-body process | 0.927--0.941 |
| fully decayed $gg\to t\bar tH$ | 20 | resonance-aware high-dimensional benchmark | 0.885--0.941 |

For the 5D process, one trained model agreed with randomized Sobol integration
at five held-out energies from 500 to 1950 GeV in two independent evaluation
runs. For the 8D process, the initial five-energy comparison agreed within the
reported uncertainties except for a provisional (2.30\sigma) fluctuation at
300 GeV. An independent higher-statistics calculation at 300 GeV gave

\[
\sigma_{\rm NF}=0.0529874\pm0.000013\ \mathrm{pb},\qquad
\sigma_{\rm Sobol}=0.0529849\pm0.0000031\ \mathrm{pb},
\]

corresponding to a $0.18\sigma$ difference and resolving that concern.

The 20D benchmark was checked more extensively: phase-space constraints,
pointwise agreement between independently generated matrix-element evaluators,
randomized Sobol integration, and explicitly surveyed/refined MadEvent
integration. Detailed numerical records are stored under `results_public/` and
summarized in `docs/VALIDATION_RESULTS.md`.

## Interpretation

The experiments support the following limited conclusion: a single
conditional marginal--copula proposal can be reused across held-out partonic
energies while maintaining accurate importance estimates and high
condition-wise ESS for the tested processes.

They do not yet establish a universal event generator or an end-to-end speedup
for arbitrary scattering calculations. In particular:

- training cost is excluded from the quoted online integration comparisons;
- the process sweep changes dimension, resonance structure, and matrix-element
  complexity simultaneously, so it is not a controlled dimension-only study;
- most reported results use one trained checkpoint with independent evaluation
  samples, rather than multiple independently trained models;
- different processes still require physically appropriate phase-space maps;
- the current formulation assumes a nonnegative integrand and does not yet
  address signed subtraction terms at higher perturbative orders; and
- the calculations are parton-level rather than full hadron-level predictions.

These boundaries are intentional: the present results validate the numerical
method and its conditional reuse without claiming more generality than has
been tested.
