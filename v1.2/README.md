# Local Information-Geometric Structure of Multi-Observer Aggregation: A KL–Fisher Bridge to Jensen–Shannon Geometry

**Author:** Vladimir Khomyakov
**ORCID:** [0009-0006-3074-9145](https://orcid.org/0009-0006-3074-9145)
**Version:** 1.2
**DOI (this version):** [10.5281/zenodo.20412703](https://doi.org/10.5281/zenodo.20412703)
**Concept DOI:** [10.5281/zenodo.20373266](https://doi.org/10.5281/zenodo.20373266)
**License:** MIT (code), CC-BY 4.0 (manuscript)

---

## Overview

This Zenodo deposit contains the manuscript and the numerical verification code
accompanying the paper:

> Khomyakov, V. (2026). *Local Information-Geometric Structure of Multi-Observer
> Aggregation: A KL–Fisher Bridge to Jensen–Shannon Geometry* (v1.2).

The paper establishes a local structural bridge between two information-geometric
frameworks for observer-dependent probability representations on finite spaces:

1. The partition-adapted parametric framework of Khomyakov (KLEO v3.24.1),
   [DOI:10.5281/zenodo.20304959](https://doi.org/10.5281/zenodo.20304959), in
   which observer entropy admits a quadratic scaling law controlled by the
   Fisher information matrix.
2. The Interference-First Reality / multi-observer JS-Fréchet aggregation
   framework of Bianchi (2026),
   [DOI:10.4236/jamp.2026.144077](https://doi.org/10.4236/jamp.2026.144077),
   in which multi-observer objectivity is identified with the Fréchet barycenter
   of observer-induced distributions in the Jensen–Shannon metric.

The three principal asymptotic statements proved in the manuscript are verified
numerically on a self-contained four-point softmax worked example:

- **Lemma 3.1 (Local Metric Bridge):** the coefficient `1/8` in front of the
  Fisher–Rao quadratic form in the local expansion of `D_JS`.
- **Theorem 4.1 (Multi-Observer Quadratic Aggregation):** the leading-order
  form of the empirical JS-Fréchet barycenter and the within-class dispersion
  law (with explicit constant `1/16` for the four-point softmax case of
  Remark 6.5).
- **Corollary 6.1:** the `O(ε^2)` coincidence of the variational Fréchet
  minimizer `p_F` and the closed-form geometric overlap estimator `p_G`.

---

## Repository Structure

```
local_information_geometric_structure_v1.2/
├── README.md
├── LICENSE
├── requirements.txt
└── scripts/
      ├── verify_js_bridge_v1.2.py
      ├── Results_real.txt
      ├── table_theorem_4_1.csv
      └── table_lemma_3_1.csv
```

### Manuscript (in this Zenodo deposit, alongside the archive)
- `local_information_geometric_structure_v1.2.pdf` — compiled PDF of the
  manuscript.
- `local_information_geometric_structure_v1.2.zip` — the reproducibility
  archive whose contents are listed below.

### Archive contents (`local_information_geometric_structure_v1.2.zip`)
- `README.md` — this file.
- `LICENSE` — MIT License (covers the code).
- `requirements.txt` — minimal Python dependencies (NumPy ≥ 1.24, SciPy ≥ 1.10).
- `scripts/verify_js_bridge_v1.2.py` — numerical verification script
  reproducing the asymptotic statements of Sections 3, 4, and 6 of the
  manuscript.
- `scripts/Results_real.txt` — console artifact produced by the verification
  script, including the side check `I_ww(0) = (1/2) I_2` and the EMPIRICAL
  SUMMARY block.
- `scripts/table_lemma_3_1.csv` — numerical table for Lemma 3.1
  (Table 1 of the manuscript).
- `scripts/table_theorem_4_1.csv` — numerical table for Theorem 4.1 and
  Corollary 6.1 (Table 2 of the manuscript).

---

## Reproducing the Numerical Verification

### Dependencies

```
numpy >= 1.24
scipy >= 1.10
```

Python ≥ 3.9 is recommended.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the verification

From the root of the extracted archive:

```bash
cd scripts
python verify_js_bridge_v1.2.py
```

The script is deterministic (a fixed seed is documented in the source). It
prints diagnostic output to the console and writes two CSV tables in the
working directory:

- `table_lemma_3_1.csv` — Experiment 1 (Lemma 3.1).
- `table_theorem_4_1.csv` — Experiment 2 (Theorem 4.1 / Corollary 6.1 /
  Remark 6.5).

The console artifact archived in this deposit (`scripts/Results_real.txt`)
corresponds to a reference run of the same script on a CPython 3.x /
NumPy ≥ 1.24 / SciPy ≥ 1.10 environment.

---

## Worked Example

The verification is carried out on the four-point softmax worked example of
Section 6.5 / Remark 6.5 of the manuscript:

- State space: `X = {1, 2, 3, 4}`, `n = 4`, `K = 3`.
- Partition: `P = {{1, 2}, {3, 4}}`, `m = 2`, `d_b = 1`, `d_w = 2`.
- Base point: `theta_0* = (0, 0, 0)`, with `p_{theta_0*} = (1/4, 1/4, 1/4, 1/4)`.
- Sufficient statistic `T : X → R^3` in the centered between/within softmax
  parameterization of Khomyakov (KLEO v3.24.1), Corollary 6.51, normalized so
  that `I_ww(0) = (1/2) I_2`.

The script also computes the full Fisher information matrix `I(theta_0*)` and
asserts the identity `||I_ww(0) - (1/2) I_2||_F < 10^{-10}` as a side check.

---

## Summary of Numerical Results

A reference run on the four-point softmax worked example yields:

- **Lemma 3.1.** Across all three perturbation directions and six decades of
  `ε`, the ratio `D_JS / [(ε^2/8) I_p(δ)]` approaches `1` with deviation
  consistent with an `O(ε^2)` relative correction; the normalized residual
  `|R| / ε^3` remains bounded across the entire grid, confirming the
  `O(ε^3)` remainder.
- **Theorem 4.1 / Remark 6.5.** The ratio
  `V_N^num / V_N^pred` with predicted leading coefficient `1/12`
  (for the symmetric configuration `α_1 = (1, 0)`, `α_2 = (0, 1)`,
  `α_3 = (-1, -1)` with `w_i = 1/3`) reaches `0.99999750` at `ε = 10^{-2.5}`.
- **Corollary 6.1.** The empirical log-log slope of `d_JS(p_F, p_G)` in `ε`
  over the tabulated grid is `≈ 2.19`, consistent with the predicted
  `O(ε^2)` rate within the resolution of the L-BFGS-B optimizer
  (`gtol = 10^{-12}`).
- **Side check.** `||I_ww(0) - (1/2) I_2||_F = 0` to machine precision.

For full numerical detail see the CSV tables and `Results_real.txt`.

---

## Citation

If you use this deposit, please cite both the manuscript and the version-pinned
DOI:

```bibtex
@misc{khomyakov_2026_20412703,
  author       = {Khomyakov, Vladimir},
  title        = {Local Information-Geometric Structure of Multi-
                   Observer Aggregation: A KL–Fisher Bridge to
                   Jensen–Shannon Geometry
                  },
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.2},
  doi          = {10.5281/zenodo.20412703},
  url          = {https://doi.org/10.5281/zenodo.20412703}
}
```

The two source frameworks on which the bridge construction depends should be
cited as:

```bibtex
@misc{khomyakov_2026_20304959,
  author       = {Khomyakov, Vladimir},
  title        = {KL-Geometric Structure of Observer Entropy: A
                   Minimal Information-Theoretic Framework
                  },
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {3.24.1},
  doi          = {10.5281/zenodo.20304959},
  url          = {https://doi.org/10.5281/zenodo.20304959}
}

@article{bianchi_2026,
  author       = {Bianchi, Michele},
  title        = {Interference-First Reality: Multi-Observer Emergence of
                  Objectivity and Time from Information Geometry},
  journal      = {Journal of Applied Mathematics and Physics},
  volume       = {14},
  number       = {4},
  pages        = {1627--1674},
  year         = {2026},
  doi          = {10.4236/jamp.2026.144077}
}
```

---

## Related Repositories and Deposits

- v1.2 source repository (this work):
  [https://github.com/Khomyakov-Vladimir/kl-fisher-js-bridge/](https://github.com/Khomyakov-Vladimir/kl-fisher-js-bridge/)
- KLEO v3.24.1 source repository:
  [https://github.com/Khomyakov-Vladimir/observer-entropy-bridge/](https://github.com/Khomyakov-Vladimir/observer-entropy-bridge/)
- Bianchi (2026) reproducibility materials:
  [https://zenodo.org/records/19023679](https://zenodo.org/records/19023679)

The v1.2 verification script `verify_js_bridge_v1.2.py` is logically distinct
from the v3.24.1 parent script `verify_bridge_theorem_v3.24.1.py` archived in
the KLEO repository: the former implements the multi-observer aggregation
experiments specific to the bridge theorem of Section 4 and the local
Jensen–Shannon expansion of Section 3, whereas the latter verifies the
single-observer KL → Fisher quadratic scaling law of Khomyakov (KLEO v3.24.1).

---

## Scope and Limitations

The bridge established in the manuscript is **local in `ε`** and **finite in
`N`**. In particular:

- It does **not** derive Bianchi's (2026) almost-sure consistency theorem
  (`N → ∞` at fixed `ε`) from Khomyakov's (KLEO v3.24.1) quadratic expansion
  (`ε → 0^+` at fixed `N`), nor conversely.
- It does **not** make a global statement: all expansions are valid in a
  compact neighborhood of the within-class neutral base point `theta_0*`.
- It does **not** assert that every empirical multi-observer ensemble admits
  a partition-adapted parametric embedding.
- It does **not** make a physical-substrate claim. The thermodynamic
  interpretation of Corollary 6.6 rests on two distinct physical-modelling
  postulates (per-channel irreversibility and inter-channel additivity),
  both flagged explicitly in the manuscript.

For a detailed discussion, see Section 7 of the manuscript.

---

## License

- **Code** (`scripts/verify_js_bridge_v1.2.py`) and the accompanying
  artifacts produced by it (`Results_real.txt`, `table_lemma_3_1.csv`,
  `table_theorem_4_1.csv`): MIT License — see the `LICENSE` file.
- **Manuscript** (compiled PDF, `local_information_geometric_structure_v1.2.pdf`):
  Creative Commons Attribution 4.0 International (CC-BY 4.0), as deposited on Zenodo.

---

## Conflicts of Interest

The author declares no conflicts of interest.

## Use of Generative AI Tools

Generative AI tools (Claude by Anthropic) were used for LaTeX formatting,
code refactoring, and stylistic editing during manuscript preparation. All
mathematical content, proofs, numerical verifications, and scientific
conclusions are the author's own and were independently verified through the
reproducible code archived in this deposit.
