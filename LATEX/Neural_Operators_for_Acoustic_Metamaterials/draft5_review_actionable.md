# draft5 actionable review

## Meta
- manuscript: `LATEX/Neural_Operators_for_Acoustic_Metamaterials/draft5.tex`
- axes: flow | concise | claims | visuals
- review_date: 2026-08-05
- scope: main text + SI in same file; no tex edits in this pass

## Summary
- Highest priority: reconcile **Huber vs NMAE** training claims (main results vs encoding ablation) and **batch 512 vs 520**.
- Remove stale **(Supplementary Information)** pointer after aliasing figure (ablation tables moved to main text).
- Soften or re-cite **finol2019eigenvalue** and **gabor1946theory** where use exceeds paper support; fix **cavallazzi2026whno** framing (Walsh–Hadamard, not wavelet).
- Cut redundancy: spectral truncation / discrete difficulty is explained three times; wavelet spatial–spectral motivation twice.
- Results section opens with “We next evaluate” and no roadmap; percentile field galleries are very long for the textual payoff.
- Align speedup wording: abstract “two to three orders… consumer-grade PC” vs conclusions “three orders… consumer grade CPU”.
- Histogram captions claim “relative” error while figure files are labeled `*_mae_*`; clarify metric.
- Design-space prose still says “polymer” while caption/body use “stiff elastomer”.
- Contribution 2 (“explain theoretically why… enables deterministic mode selection”) overclaims relative to Gabor citation + empirical ablation.
- Encoding formulas match code/runs; Methodology 3-channel wavelet focus is fine (do-not-change).

## Issues

### ISS-001
- axis: claims
- severity: blocker
- location: `ssec:training_procedure` (~L501, L583); `ssec:band_wavevector_encoding` ablation para (~L885); SI Loss Criterion (~L1019)
- claim_or_passage: "Huber loss … was used for all reported results" vs ablation "NMAE training objective"
- problem: Main Training Procedure and SI state Huber for all reported results. Encoding ablation (and its E8 runs) used NMAE. Readers cannot tell which objective produced the displacement/dispersion figures.
- evidence: Ablation text L885; training run configs `…_insin_260723` / `…_inconst_260725` use `loss: nmae`. Main text L583 claims Huber + batch 512 + 12 epochs for all article results.
- action: Split claims. State that the primary wavelet model reported in Displacement/Dispersion/Boundary subsections was trained with Huber (if true). State that the encoding ablation used NMAE under matched hyperparameters. Delete or qualify “all reported results” wherever it is false.
- acceptance: Grep for “all reported results” / “Huber” / “NMAE training”; no remaining claim that one loss covers both primary and ablation results unless both actually used it.

### ISS-002
- axis: claims
- severity: major
- location: `ssec:training_procedure` (~L583, L611); ablation (~L885); SI `tab:training_hyperparameters`
- claim_or_passage: batch size 512 (training/SI) vs batch size 520 (ablation)
- problem: Inconsistent batch size across the manuscript for “matched” or “selected” training settings.
- evidence: L583/L611 and SI table italicize 512; L885 says 520 (matches encoding E8 `resolved_config.json`).
- action: Either (a) report ablation batch 520 and primary-model batch 512 as distinct, or (b) unify if one number is wrong. Do not call settings “the same” across runs that differ in batch size without noting it.
- acceptance: Single consistent story: primary config numbers match SI table; ablation paragraph lists only true shared settings.

### ISS-003
- axis: flow
- severity: major
- location: `ssec:input_wavelet_encoding` (~L450)
- claim_or_passage: "…Algorithm~\ref{alg:2d_gabor_embedding} (Supplementary Information)."
- problem: Parenthetical implies the similarity analysis or algorithms live only in SI. Algorithms are in SI, but the similarity figure is in main Methods. After moving ablation tables out of SI, this stale SI pointer confuses readers.
- evidence: Figure `\ref{fig:wavevector_encoding_similarity}` is immediately below in main text; no SI subsection remains for that matrix.
- action: Replace “(Supplementary Information)” with a pointer to Algorithm~\ref{alg:2d_gabor_embedding} only, or “see Algorithms in the Supplementary Information” without implying the heatmap is SI.
- acceptance: No dangling SI reference for content that is in the main text.

### ISS-004
- axis: claims
- severity: major
- location: Introduction (~L242); related-work sentence citing `finol2019eigenvalue`
- claim_or_passage: Eigenvalue PDEs "require simultaneous recovery of an unknown eigenparameter and one of several valid eigenmodes~\cite{finol2019eigenvalue,hussein2014dynamics}"
- problem: `finol2019eigenvalue` is a CNN surrogate for phononic *eigenvalues*, not an argument that operator learning fails because of multi-mode selection / unknown eigenparameter framing. `hussein2014dynamics` supports Bloch eigenproblems but not the ML formulation gap.
- evidence: Finol et al. abstract (bib): predict eigenvalues of phononic crystals with CNNs. Does not develop the “one of several eigenmodes” operator-learning obstruction as used here.
- action: Keep Finol for “CNN eigenvalue surrogates in phononics.” Move the multi-mode / nonunique map claim to uncited author reasoning or a more precise cite. Pair Hussein only with Bloch/phononic physics sentences.
- acceptance: Each cite’s local sentence matches what that paper actually does.

### ISS-005
- axis: claims
- severity: major
- location: Contributions item 2 (~L251); wavelet Methods (~L412); Results encoding (~L883); Conclusions (~L941)
- claim_or_passage: "explain theoretically why their spatial--spectral structure enables deterministic mode selection in FNOs" / Gabor citations for aliasing and FNO branch utilization
- problem: `gabor1946theory` supports jointly localized space–frequency atoms. It does not prove FNO mode selection or that finite-grid encodings will not alias. Mode selection is an empirical claim (ablation). Aliasing check is an empirical cosine-similarity experiment.
- evidence: Gabor 1946 bib abstract; citation at L450 for aliasing is a stretch.
- action: Soften contribution 2 to “motivate” or “argue.” Cite Gabor only for space–frequency atom construction. Remove Gabor cite from the aliasing sentence or replace with a signal-processing / DFT aliasing reference (or no cite).
- acceptance: No sentence claims Gabor theory alone establishes FNO deterministic mode selection or non-aliasing of the IBZ encodings.

### ISS-006
- axis: claims
- severity: minor
- location: Results continuous-vs-discrete (~L744); boundary subsection (~L846); Conclusions (~L943)
- claim_or_passage: spectral limitations citing `rahaman2019spectral,qin2024spectralfno,cavallazzi2026whno`
- problem: Rahaman supports generic NN spectral bias (supports). Qin supports FNO Fourier/spectral bias (supports). Cavallazzi is Walsh–Hadamard NO for discontinuous coefficients, useful for “Fourier struggles at interfaces,” but it is not a wavelet method and should not be read as endorsing wavelet encodings.
- evidence: `cavallazzi2026whno` bib abstract (WHNO vs FNO for discontinuous PDEs).
- action: Keep Cavallazzi for Gibbs/interface / discontinuous-coefficient Fourier difficulty. Do not imply it studies wavelets. Optionally drop it from sentences about “wavelet encodings mismatched to interfaces.”
- acceptance: Local wording around Cavallazzi mentions discontinuous coefficients / interfaces / Fourier limits, not wavelets.

### ISS-007
- axis: claims
- severity: minor
- location: Abstract (~L192); Conclusions (~L943)
- claim_or_passage: Abstract "two to three orders of magnitude … consumer-grade PC" vs Conclusions "three orders of magnitude … consumer grade CPU"
- problem: Speedup magnitude and hardware wording disagree.
- evidence: Abstract L192; Conclusions L943 (~1 ms vs ~1 s implies ~10^3).
- action: Unify to one magnitude phrase and one hardware phrase (prefer Conclusions’ three orders + consumer grade CPU if 1 ms vs 1 s is the measured pair).
- acceptance: Abstract and Conclusions use identical speedup and hardware wording.

### ISS-008
- axis: flow
- severity: major
- location: `\section{Results and Discussion}` (~L617–L632)
- claim_or_passage: Empty section lead-in; first subsection starts "We next evaluate"
- problem: “We next” has no antecedent. Results has no roadmap tying fields → dispersion → boundary → encodings.
- evidence: Commented checklist L618–L626; live text jumps to Displacement Field Prediction.
- action: Add 2–4 sentence Results roadmap. Change “We next evaluate” to “We evaluate” or “We first evaluate.”
- acceptance: Results opens with an explicit plan; no orphan “next.”

### ISS-009
- axis: concise
- severity: major
- location: `ssec:Continuous Versus Discrete` (~L744); `ssec:boundary_length_vs_loss` (~L846–L857); Conclusions (~L943)
- claim_or_passage: Repeated explanation that truncated Fourier / spectral bias hurts discrete interfaces and higher bands
- problem: Same mechanistic story appears in Continuous Versus Discrete, again at length in Boundary Length, and again in Conclusions.
- evidence: Parallel wording on truncated modes, broadband spectra, Gibbs-type artifacts.
- action: Keep full mechanism once (prefer Boundary Length). In Continuous Versus Discrete, keep 1–2 sentences + forward ref to `\ref{ssec:boundary_length_vs_loss}`. Shorten Conclusions to one sentence pointing back to Results.
- acceptance: Mechanism explained in detail in only one Results subsection.

### ISS-010
- axis: concise
- severity: major
- location: `ssec:input_wavelet_encoding` (~L410–L412); `ssec:band_wavevector_encoding` (~L883)
- claim_or_passage: Wavelets rich in space and spectrum; both FNO branches; superior to constant/sinusoidal
- problem: Motivation for wavelets is stated in Methods and restated as the punchline of Results encoding, before the tables quantify it.
- evidence: L412 and L883.
- action: Methods: define constructions + point to Results for why superior. Results: keep comparative formulas + ablation + short “both branches” interpretation. Avoid repeating the full motivation paragraph.
- acceptance: Full “why wavelets win” narrative appears once in Results Encoding (or once in Methods), not both at full length.

### ISS-011
- axis: concise
- severity: minor
- location: Displacement Field Prediction continuous + discrete (~L634–L739)
- claim_or_passage: Six large figure environments (input+output at p25/p50/p75 × 2 geometry types)
- problem: High page cost; continuous and discrete closing paragraphs are nearly copy-paste. Many captions repeat “first row target, second prediction, third difference.”
- evidence: L634–L739 parallel structure.
- action: Prefer one multi-panel figure per geometry type (3 percentiles) or move p25/p75 to SI and keep median + one weak/strong in main. Deduplicate closing paragraphs.
- acceptance: Main text does not repeat the same caption boilerplate six times; page count of field galleries reduced or justified in one consolidated figure.

### ISS-012
- axis: visuals
- severity: major
- location: `fig:disp_loss_histograms` (~L763); `fig:freq_loss_histograms` (~L838)
- claim_or_passage: Captions say "relative prediction error" while image paths are `displacement_loss_histogram_mae_*` / `freq_loss_histogram_mae_*`
- problem: Caption metric may not match plotted quantity (MAE vs NMAE/relative).
- evidence: `\includegraphics{...mae_c_test.png}` vs caption “relative prediction error” and body text citing \(10^{-2}\)–\(10^{-4}\) relative errors.
- action: Open the plot-generation notebook/script; confirm y-axis metric. Align caption, body, and filename language (MAE vs NMAE vs relative).
- acceptance: Caption metric name matches the plotted statistic and the numbers quoted in prose.

### ISS-013
- axis: visuals
- severity: minor
- location: `fig:example_geometries` caption (~L293); Design Space body (~L286–L288)
- claim_or_passage: Caption uses stiffer/softer; body still says "pure polymer" / "polymer and steel"
- problem: Phase naming inconsistent after elastomer rename (table headers still `polymer`).
- evidence: L269 stiff elastomer; L286 pure polymer; L275 `E_{\text{polymer}}`.
- action: Use elastomer consistently in prose, or one sentence: “elastomer (labeled polymer in Table…)”. Optionally rename table macros to `elastomer` if desired.
- acceptance: No unexplained polymer/elastomer flip for the soft phase.

### ISS-014
- axis: flow
- severity: minor
- location: `ssec:input_wavelet_encoding` opener (~L410)
- claim_or_passage: "This information is given in the form of a wavevector embedding" covering band and wavevector
- problem: “Wavevector embedding” underspecifies band encoding.
- evidence: L410 vs later 1D band + 2D wavevector algorithms.
- action: Change to “wavelet embeddings of the wavevector and band index.”
- acceptance: Opener names both conditioning quantities.

### ISS-015
- axis: flow
- severity: nit
- location: `ssec:training_procedure` (~L499)
- claim_or_passage: "paired inputs and outputs described in Section \ref{sec:Methodology}"
- problem: Points to the whole Methodology section instead of Dataset Construction.
- evidence: L499.
- action: Cite `\ref{ssec:dataset_construction}` (and optionally encodings).
- acceptance: Training opener points to the tensorization subsection.

### ISS-016
- axis: claims
- severity: minor
- location: Introduction (~L236); Cummer/Liao cluster
- claim_or_passage: "sound attenuation, vibration isolation, and wave focusing~\cite{cummer2016controlling,liao2021acoustic}"
- problem: Reviews strongly support attenuation/focusing/manipulation; vibration isolation is a weaker fit to those specific reviews.
- evidence: Bib abstracts for Cummer/Liao.
- action: Drop “vibration isolation” or support it with a more targeted cite.
- acceptance: Listed applications are attested in the cited reviews.

### ISS-017
- axis: claims
- severity: minor
- location: Continuous Versus Discrete (~L744)
- claim_or_passage: "continuous geometries more often contain large regions of near-zero displacement, which inflate relative error through a small denominator"
- problem: As written, this explains why continuous relative error could look *worse*, but the subsection argues continuous performs *better*. Confusing / possibly misplaced.
- evidence: L744 in a paragraph about poorer discrete performance.
- action: Delete, move to a caveat about interpreting NMAE, or clarify that this effect acts against the observed continuous advantage (i.e. continuous still wins despite that bias).
- acceptance: Sentence no longer undercuts the discrete-harder claim without explanation.

### ISS-018
- axis: claims
- severity: nit
- location: Training (~L611) vs ablation (~L885)
- claim_or_passage: scheduler step size fixed at 4 vs ablation "per-epoch decay 0.9"
- problem: Primary text fixes step size 4; encoding E8 configs used step_size 1 with gamma 0.9. Another “matched hyperparameters” subtlety.
- evidence: L611; ablation L885; encoding run `resolved_config.json` step_size 1.
- action: When listing ablation hyperparameters, include scheduler step size explicitly if it differs from the primary model.
- acceptance: Ablation hyperparameter list matches the actual ablation configs.

### ISS-019
- axis: visuals
- severity: minor
- location: Dispersion percentile figures (~L769–L819)
- claim_or_passage: Six dispersion panels + histograms; body says average error below 1% on encoded eigenfrequency
- problem: Useful, but captions do not state that curves are IBZ path restrictions (body does). Filename NMAE/NMSE in paths may confuse with displacement NMAE.
- evidence: L794 path restriction in body only.
- action: Put IBZ path restriction into dispersion figure captions. Clarify “encoded eigenfrequency” error definition once (log-encoded channel vs physical Hz).
- acceptance: Captions self-contained on path sampling and error definition.

### ISS-020
- axis: concise
- severity: nit
- location: Abstract (~L192) vs Intro contributions vs Conclusions
- claim_or_passage: Repeated speedup + discontinuity + wavelet matching themes
- problem: Appropriate thematic repetition, but abstract “theoretical motivation” echoes contribution 2 overclaim (ISS-005).
- evidence: Abstract L192.
- action: Align abstract wording with softened contribution 2 (“motivate” / “argue” + experimental evidence).
- acceptance: Abstract does not promise a theory that the paper does not deliver.

## Citation ledger
Format: `key | used_for (approx locus) | verdict | note`

- `cummer2016controlling` | AM control/attenuation/focusing (Intro) | partial | Weak on vibration isolation
- `liao2021acoustic` | AM review cluster (Intro) | partial | Same
- `kushwaha1993acoustic` | Phononic band structure / Bloch (Intro, FEA) | supports | Foundational
- `hussein2014dynamics` | Phononic dynamics / Bloch FEA context | supports | Do not use for ML multi-mode formulation gap
- `Zhang2024uq` | Costly repeated bandgap/FEA evaluations, UQ | supports | Authors’ prior UQ/FEA sampling paper
- `kovachki2023neural` | Neural operator UA / function-space maps | supports |
- `lu2021deeponet` | Operator learning / DeepONet | supports | Decorative relative to FNO focus but accurate
- `li2021fno` | FNO architecture / Fourier layers | supports |
- `wang2021pidon` | Physics-informed operators (Intro, future work) | supports |
- `finol2019eigenvalue` | Eigenvalue PDEs / CNN surrogates | partial | OK for CNN eigenvalue prediction; mismatch for multi-mode operator gap
- `liu2019dispersion` | Dispersion-from-params NN | supports |
- `Ogren2024gpr` | GPR dispersion surrogate | supports |
- `wagner2024neural` | NO transmission-loss spectra | supports |
- `liu2026hybridfno` | Wavevector-conditioned band FNO | supports |
- `liu2023deeplearning` | DL design review | supports | Broad
- `jin2022intelligent` | ML metamaterial design review | supports | Broad
- `muhammad2022ml` | ML phononic review | supports | Broad
- `CHEN2022101895` | Interpretable bandgap design (Intro; removed from Conclusions future work) | supports | Still in Intro related work
- `Bastawrous2025` | Interpretable hierarchical design (Intro) | supports | Still in Intro related work
- `rahaman2019spectral` | Spectral bias low-frequency preference | supports | Generic NN, not FNO-specific
- `qin2024spectralfno` | FNO spectral / Fourier bias | supports |
- `cavallazzi2026whno` | Interfaces / discontinuous coeffs vs Fourier | partial | Walsh–Hadamard NO; not wavelets
- `gabor1946theory` | Space–frequency atoms | partial | Supports encoding construction; not aliasing proof or FNO mode selection
- `loshchilov2019adamw` | AdamW optimizer | supports |
- `kingma2015adam` | Adam baseline | supports |

## Visual ledger
Format: `label | verdict | note`

- `fig:example_geometries` | keep + edit-caption/body | Black/white mapping good; align polymer/elastomer naming (ISS-013)
- `fig:FEA_full_process` | keep | Useful pipeline; caption dense but OK
- `fig:FNO_architecture` | keep | Standard; caption long
- `fig:1D_wavelet_encoding` | keep | Essential for Methods
- `fig:2D_wavelet_encoding` | keep | Essential for Methods
- `fig:wavevector_encoding_similarity` | keep | Supports non-aliasing claim; fix SI pointer (ISS-003)
- `fig:dataset_construction` | keep | Clarifies 3-channel assembly
- `fig:c_p25_*` … `fig:c_p75_*` | keep or merge | Consider consolidating (ISS-011)
- `fig:b_p25_*` … `fig:b_p75_*` | keep or merge | Same; p25 caption about Huber absolute vs relative is valuable—retain somewhere
- `fig:disp_loss_histograms` | edit-caption | Resolve MAE vs relative (ISS-012)
- `fig:dispersion_c_percentiles` | keep + edit-caption | Add IBZ path note (ISS-019)
- `fig:dispersion_b_percentiles` | keep + edit-caption | Same
- `fig:freq_loss_histograms` | edit-caption | Same metric issue as disp histograms (ISS-012)
- `fig:boundary_vs_mae_by_band` | keep | Strongest empirical support for interface claim
- `tab:parameter_ranges` | keep | Values match code steel/elastomer constants
- `tab:encoding_ablation_median` | keep | Headers OK after recent edit
- `tab:encoding_ablation_mean` | keep | Same
- `tab:training_hyperparameters` (SI) | keep | Italic selected values; reconcile with ablation 520/NMAE (ISS-001/002)
- `fig:eigenfrequency_encoding` (SI) | keep | Separate encode–decode demo; ensure not confused with input band/k encodings
- `fig:eigenfrequency_decode_error` (SI) | keep | Same

## Do-not-change
- Methodology subsection order (Design → FEA → FNO → Wavelets → Dataset → Training).
- Methodology focus on **wavelet / 3-channel** FNO I/O (do not rewrite Methods around uniform 4-channel ablation).
- SI **Algorithms for Input Wavelet Encoding** wavelet-only (no requirement to add uniform/sinusoidal algorithms).
- Encoding **formulas** in Results for constant and sinusoidal (match `NO_utilities.py` / E8 runs).
- Softened uniform-encoding language (fragile / high LR / table for best case).
- Conclusions emphasis on metamaterial design-loop acceleration and future work on symmetries / multimaterial spaces.
- Material table numerical values \(E=10^8,\,2\times10^{11}\), \(\rho=1200,8000\), \(\nu=0.45,0.3\).
