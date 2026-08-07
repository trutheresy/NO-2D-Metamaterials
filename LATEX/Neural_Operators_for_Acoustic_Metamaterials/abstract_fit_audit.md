# Abstract fit audit vs suggested citation anchors

Legend: **good** = abstract supports intended use; **partial** = related but cite carefully / differentiate; **weak** = distant; **mismatch** = should not cite for that claim.

Abstracts injected for **31** entries.

| Key | Fit | Suggested anchor | Audit note |
|---|---|---|---|
| `cummer2016controlling` | good | Intro L241 — acoustic metamaterials context | Canonical review of controlling sound with acoustic metamaterials; matches intro motivation. |
| `liao2021acoustic` | good | Intro L241 — acoustic MM review | Broad AM theories/structures/applications review; suitable alongside Cummer. |
| `hussein2014dynamics` | good | Intro L241 / FEA §2.3 — phononic dynamics, band structure, FEM | Core phononics dynamics review emphasizing band structure and numerical methods; excellent fit. |
| `kushwaha1993acoustic` | good | Intro L241 — classic phononic band structure | First full elastic phononic band-structure calculation; historical foundation cite. |
| `bilal2011ultrawide` | partial | Design space / FEA — phononic bandgaps | About optimizing large bandgaps in silicon/void PnCs, not operator learning or displacement fields. OK as phononic-bandgap physics example, not as ML prior. |
| `celli2019bandgap` | weak | Optional phononics context — bandgap widening | Experimental stubbed-plate locally resonant bandgaps with disorder/grading. Distant from FNO eigenmode learning; keep only if discussing bandgap phenomenology. |
| `ZHU2023107013` | partial | FEA §2.3 — Bloch / band-structure computation | ROM for faster Bloch band-structure FEM, not ML. Fits as related FEA/Bloch computational method, not as neural-operator prior. |
| `mazzotti2023bio` | weak | Optional hierarchical elastic MM | Hierarchical elastic metamaterials / multiple bandgap mechanisms. Not about neural operators; only loosely related via elastic MM hierarchy. |
| `CHEN2022101895` | good | Intro / related work — Brinson-group interpretable MM design | Interpretable ML for bandgap design of 2D metamaterials; same group/theme, complementary (design rules vs operator surrogate). |
| `Bastawrous2025` | good | Intro / related work — hierarchical phononic templates | Interpretable ML hierarchical bandgap design; same collaboration line as Chen2022; complementary prior. |
| `Zhang2024uq` | partial | Intro L235 — costly repeated FEA / UQ | First-author prior on acoustic MM dispersion under uncertainty using PCE — motivates expensive repeated FEA, but paper is UQ not neural operators. Cite carefully as prior acoustic-MM FEA work, not as NO method. |
| `Ogren2024gpr` | good | Intro related-work — surrogate for dispersion | GPR surrogate for dispersion relations in architected materials; closest author-linked surrogate prior. Matches 'surrogate for dispersion' use; differentiate from FNO field+eigenmode prediction. |
| `li2021fno` | good | Intro L237 / §2.1 — FNO definition | Seminal FNO paper; exact match. |
| `kovachki2023neural` | good | Intro L237 / §2.1 universal approximation todo | Neural operator theory + universal approximation; exact match for theory todo. |
| `lu2021deeponet` | good | Intro L237 — operator learning lineage | DeepONet foundational operator-learning paper; correct for broader NO context. |
| `wang2021pidon` | partial | Intro L237 optional — physics-informed operators | Physics-informed DeepONet for parametric PDEs. Related to operator learning but PI training is not used in this manuscript; optional only. |
| `rahaman2019spectral` | good | Results L728 — spectral bias | Classic spectral/low-frequency bias of NNs; supports discrete/high-frequency difficulty argument. |
| `qin2024spectralfno` | good | §2.1 / Results — FNO spectral limitations | Analyzes FNO from a spectral perspective (Fourier parameterization bias); supports truncated-mode discussion. |
| `cavallazzi2026whno` | good | Results L728 / boundary subsection — discontinuities vs Fourier | Explicitly argues Gibbs / discontinuous coefficients hurt FNO; strong support for binary-geometry claim. |
| `liu2019dispersion` | good | Intro related-work — NN predicting dispersion | Early NN prediction of 1D PnC dispersion from scalar parameters; correct contrast prior (not field operators / eigenmodes). |
| `liu2023deeplearning` | good | Intro related-work — DL for PnC/EM design review | Review of DL for phononic crystals and elastic metamaterials; fits survey cite. |
| `jin2022intelligent` | good | Intro related-work — ML phononic design review | Review of ML-driven on-demand phononic metamaterial design; fits survey cite (somewhat overlapping muhammad/liu reviews). |
| `muhammad2022ml` | good | Intro related-work — ML/DL phononics review | ML/DL in PnC and metamaterials review; good survey cite; redundant if both Jin and Liu2023 kept. |
| `wagner2024neural` | good | Intro related-work — NOs for acoustic MM | FNO/DeepONet for sonic-crystal transmission-loss spectra; closest NO+acoustic-MM prior. Cite and differentiate (TL curve vs Bloch displacement eigenmodes). |
| `liu2026hybridfno` | good | Intro related-work — Hybrid FNO band structures | Wavevector-conditioned Hybrid FNO for phononic band structures; very close prior. Cite and differentiate (eigenfrequencies vs full complex displacement fields + wavelet mode selection). |
| `finol2019eigenvalue` | good | Intro L239 — ML for eigenvalue problems | Deep CNNs for eigenvalue problems in mechanics; supports eigenvalue-PDE ML context (not FNO/operators specifically). |
| `gabor1946theory` | good | §2.4 — Gabor wavelet encodings | Foundational Gabor time-frequency atoms; appropriate for Gabor encoding methods. |
| `mallat2009wavelet` | good | §2.4 — wavelet spatial-spectral representations | Standard wavelet monograph; appropriate methods cite (book blurb is descriptive, not a journal abstract). |
| `mallat2012scattering` | partial | §2.4 / encoding discussion — optional scattering | Group-invariant scattering networks; related wavelet-CNN theory but stronger than needed unless you invoke scattering stability. Prefer Mallat book + Gabor for core methods. |
| `loshchilov2019adamw` | good | Training L564 — AdamW | Defines AdamW / decoupled weight decay; exact match. |
| `kingma2015adam` | good | Training L564 — Adam baseline | Original Adam optimizer; fine as baseline contrast to AdamW. |

## Summary counts

- **good**: 24
- **partial**: 5
- **weak**: 2

## Recommendations

1. Prefer **good** cites for the corresponding anchors.
2. For **partial** cites (`Zhang2024uq`, `wang2021pidon`, `ZHU2023107013`, `bilal2011ultrawide`, `mallat2012scattering`): keep only with differentiating wording.
3. Consider dropping or demoting **weak** cites (`celli2019bandgap`, `mazzotti2023bio`) unless you add matching context sentences.
4. Among ML-phononics reviews, avoid stacking all three of `liu2023deeplearning`, `jin2022intelligent`, `muhammad2022ml` — pick 1–2.
5. Closest competitors to differentiate explicitly: `Ogren2024gpr`, `wagner2024neural`, `liu2026hybridfno`.
