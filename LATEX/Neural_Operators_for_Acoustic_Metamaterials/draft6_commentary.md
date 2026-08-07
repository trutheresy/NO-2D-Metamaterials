# draft6 commentary

## Meta
- manuscript: `LATEX/Neural_Operators_for_Acoustic_Metamaterials/draft6.tex`
- axes: coherence | flow | clarity | concision
- review_date: 2026-08-06
- scope: commentary tracking open issues; citation-locus fixes applied in `draft6.tex`
- prior review: `draft5_review_actionable.md` (historical only)
- resolved this pass: ISS-001 (histogram MAE vs relative—accepted); ISS-003 (near-zero relative-error sentence removed from Continuous Versus Discrete); ISS-009 (field-gallery page cost—nonissue)

## Summary
- Citation locus fixes applied (Finol/Hussein off multi-mode L242; Gabor off aliasing L450; Cavallazzi off L693, kept on L795).
- Remaining: FEA vs FNO geometry-channel wording (ISS-001); spectral / wavelet prose duplication (ISS-002/003); SI Gamma \(0.1\) vs runs \(0.9\) (ISS-004); caption/flow/SI nits (ISS-005–010).

## Issues

### ISS-001
- axis: coherence
- severity: major
- location: Design Space (~L269); Wave Prop (~L320–L326); FNO (~L350)
- claim_or_passage: Design Space / FEA: three material-property channels \((E,\rho,\nu)\). FNO input: “a geometry channel” + wavevector + band encodings
- problem: Readers cannot tell how the single FNO geometry channel relates to the three FEA material-property channels.
- evidence: Explicit “three-channel material representation” for FEA vs “geometry channel” for the operator.
- action: Clarify that channel 0 of the FNO input is a spatial indicator of where elastomer vs steel (or continuous phase fraction) is located; the material constants \((E,\rho,\nu)\) are not separate FNO input channels—they are implied by that location (via the fixed two-phase / GP material assignment used to build FEA labels). One sentence tying Design Space ↔ FNO I/O is enough; do not invent a three-channel FNO input.
- acceptance: Readers understand geometry channel = material location map; properties are table/lookup-implied, not independent FNO channels.

### ISS-002
- axis: concision
- severity: major
- location: see redundancy map below
- claim_or_passage: Truncated-Fourier / broadband interface / Gibbs / jump-discontinuity mechanism
- problem: Same physical/architectural explanation is written out at similar length in three places.
- redundancy map (where the repeat lives):
  1. **Results → Continuous Versus Discrete Performance** (`\subsubsection{Continuous Versus Discrete Performance}`, ~L690–L693): full mechanism—smooth vs jump discontinuities, truncated Fourier \(\mathcal{G}\), sharp boundaries, broadband high-wavenumber content, fixed \(32\times32\) grid, mismatch to smooth wavelets (cites Rahaman/Qin).
  2. **Results → Boundary Length Versus Prediction Loss** (`\subsubsection{...}` / `\label{ssec:boundary_length_vs_loss}`, ~L795–L806): longest, most complete retelling—truncated spectral modes after FFT, continuous energy in low–moderate \(k\), binary discontinuities → Gibbs / broadband content poorly captured on retained modes; then boundary-length Spearman trends and band-index MAE rise as empirical support. Also cites Cavallazzi on Gibbs.
  3. **Conclusions** (~L892): short but still re-states the same spectral perspective (error vs interface length and band; truncated Fourier vs sharp boundaries / oscillating modes; Rahaman/Qin/Cavallazzi).
- action: Keep the full mechanism once—prefer Boundary Length (~L795), where the figure evidence lives. In Continuous Versus Discrete (~L693), keep 1–2 sentences + forward ref to `\ref{ssec:boundary_length_vs_loss}`. In Conclusions (~L892), one sentence pointing back to Results, not a third mini-lecture.
- acceptance: Detailed truncated-Fourier / interface mechanism appears in only one Results subsection; the other two sites are pointers or one-liners.

### ISS-003
- axis: concision
- severity: major
- location: see redundancy map below
- claim_or_passage: “Why wavelets win” (rich in space and spectrum; both FNO branches; superior to constant/sinusoidal)
- problem: Comparative / mechanistic wavelet narrative still appears in Methods and again as the Results Encoding punchline.
- redundancy map (where the repeat lives):
  1. **Methods → Input Wavelet Encoding** (`\subsection{Input Wavelet Encoding}` / `\label{ssec:input_wavelet_encoding}`, ~L408–L412): opener already points to Results (`Section~\ref{ssec:band_wavevector_encoding}`), but ~L412 still gives the full “why”: rich variation in spatial and Fourier domains → both FNO branches interact → better learning (plus Gabor cite). Algorithms + similarity figure follow in the same subsection.
  2. **Results → Band and Wavevector Encoding** (`\label{ssec:band_wavevector_encoding}`, ~L809–L834): after constant/sinusoidal formulas and failure modes (~L830 grainy / weak Fourier branch), ~L832 retells Gabor space–spectrum tradeoff, both branches usable, encodings should match operator structure—same punchline as Methods L412, at similar length, immediately before the ablation tables (~L834).
- note: Continuous Versus Discrete (~L691) also previews encoding failure (grainy / mode collapse) before the Encoding subsection—related flow issue tracked as ISS-007.
- action: Methods: constructions + similarity check + keep the existing pointer to Results; trim or cut the L412 “both branches / better outcomes” paragraph so Methods does not argue the ablation conclusion. Results: keep formulas, failure modes, short branch-utilization interpretation (~L832), and tables.
- acceptance: Full “why wavelets win” narrative is not repeated at full length in both Methods and Results.

### ISS-004
- axis: clarity / coherence
- severity: minor
- location: SI `tab:training_hyperparameters` (~L958); Training (~L558); ablation (~L834); primary run configs `…_260711`, `…_insin_260723`
- claim_or_passage: Gamma row italicizes \(0.1\); prose/ablation use per-epoch decay \(0.9\)
- problem: Same class of “selected italic ≠ reported run” inconsistency previously fixed for batch, step size, and WD.
- evidence: Table `\textit{0.1}`; ablation “per-epoch decay \(0.9\)”; configs `gamma: 0.9`.
- action: Italicize \(0.9\) in the Gamma row (add \(0.9\) as a candidate column value if needed). Ensure Training does not imply a different scheduler gamma.
- acceptance: SI italic Gamma matches reported primary and ablation configs.

### ISS-005
- axis: clarity / visuals
- severity: minor
- location: `fig:wavevector_encoding_similarity` caption (~L475) vs body (~L450–L470)
- claim_or_passage: Caption still says “mean-centered \(\log_{10}|\mathrm{FFT}|\) spectra” without \(S\) or \(\widehat{\psi}\)
- problem: Body now defines \(\psi_{\mathbf{k}}\), \(\widehat{\psi}_{\mathbf{k}}\), \(\mathbf{s}_{\mathbf{k}}\), \(S(\mathbf{k},\mathbf{k}')\). Caption does not use that notation.
- evidence: Caption L475 vs equations L451–L468.
- action: Update caption to mention pairwise cosine similarity \(S(\mathbf{k},\mathbf{k}')\) of mean-centered entrywise \(\log_{10}|\widehat{\psi}_{\mathbf{k}}|\).
- acceptance: Caption terminology matches the body equations.

### ISS-006
- axis: clarity / visuals
- severity: minor
- location: Dispersion figure captions (~L740, L767); body (~L743, L770)
- claim_or_passage: Captions omit IBZ path restriction; body states horizontal/vertical/diagonal traversals
- problem: Captions are not self-contained. Filename NMAE/NMSE tags may also confuse readers with displacement NMAE.
- evidence: Body “restricted to the horizontal, vertical, and diagonal IBZ traversals.”
- action: Add IBZ-path restriction to both dispersion figure captions. Optionally one sentence defining “encoded eigenfrequency” error (log-encoded channel vs physical Hz).
- acceptance: Captions state path sampling; error definition is clear once in Dispersion.

### ISS-007
- axis: flow
- severity: minor
- location: Continuous Versus Discrete (~L691); Band and Wavevector Encoding (~L809+)
- claim_or_passage: Continuous-vs-Discrete already blames constant/sinusoidal encodings before the Encoding subsection
- problem: Encoding failure modes are asserted before the dedicated Encoding section and tables, so the later section partly re-argues settled ground.
- evidence: L691 vs L809–L834.
- action: In Continuous Versus Discrete, keep a short forward pointer (“encoding comparisons in Section~\ref{ssec:band_wavevector_encoding}”) without the full grainy/mode-collapse claim, or move that sentence into Encoding.
- acceptance: Encoding performance claims are concentrated in the Encoding subsection (with tables).

### ISS-008
- axis: coherence
- severity: minor
- location: SI `ssec:wavelet_decoding` (~L1139–end)
- claim_or_passage: Encode–decode fidelity experiment for positive scalars / eigenfrequency-style wavelet round-trip
- problem: Never referenced from the main Methodology or Results. Readers may confuse it with *input* band/k encodings. Feels like an unlinked SI appendix.
- evidence: No `\ref{ssec:wavelet_decoding}` / `\ref{fig:eigenfrequency_encoding}` in main text (grep).
- action: Either add one main-text forward pointer (e.g. in Dataset/Training when discussing log-encoded eigenfrequency channel), or move/cut if not needed for the paper’s claims.
- acceptance: SI decoding subsection is either cited from main text with a clear “not the conditioning encodings” note, or removed.

### ISS-009
- axis: clarity
- severity: nit
- location: Training Procedure (~L558)
- claim_or_passage: “an abrupt, irreversible loss spikes”
- problem: Grammar agreement (singular article + plural noun).
- evidence: L558.
- action: “an abrupt, irreversible loss spike” or “abrupt, irreversible loss spikes.”
- acceptance: Grammatically correct.

### ISS-010
- axis: coherence
- severity: nit
- location: Training (~L530); ablation (~L834); inference practice (`…_260711_E12`)
- claim_or_passage: “12 epochs. All results reported… this configuration” vs ablation E8 fair compare; primary training continued past 12; figures often from E12
- problem: Mild tension between “12 epochs for all results” and (a) ablation E8 reporting, (b) longer training runs / best checkpoints.
- evidence: L530; L834 epoch-8 note; run `260711` summary best epoch 26.
- action: Qualify: primary Results figures use the epoch-12 checkpoint of the NMAE wavelet model; ablation uses epoch 8 under matched budget; longer training was explored but not required for the reported tables/figures (if true). Avoid absolute “all results / 12 epochs” if false.
- acceptance: Epoch claims match the checkpoints behind figures and tables.

## Citation ledger
Format: `key | locus | verdict | note`

- `finol2019eigenvalue` | L242 CNN clause | supports | Multi-mode sentence left uncited (fixed)
- `hussein2014dynamics` | L236, L304 | supports | Removed from former multi-mode cite (fixed)
- `gabor1946theory` | L412 (and L832/L890 OK) | supports | Removed from L450 aliasing (fixed)
- `cavallazzi2026whno` | L795 (Gibbs); L892 optional | supports | Removed from L693 (fixed)
- `cummer2016controlling`, `liao2021acoustic` | L236 | supports | No change

## Visual ledger (edit / watch)
Format: `label | verdict | note`

- `fig:wavevector_encoding_similarity` | edit-caption | Align with \(S(\mathbf{k},\mathbf{k}')\) / \(\widehat{\psi}\) (ISS-005)
- `fig:disp_loss_histograms`, `fig:freq_loss_histograms` | keep | MAE vs relative accepted as resolved
- `fig:dispersion_*_percentiles` | edit-caption | IBZ path restriction (ISS-006)
- `fig:c_p*_*`, `fig:b_p*_*` | keep | Page-cost consolidation treated as nonissue
- `fig:eigenfrequency_encoding` (SI) | keep + link or cut | Orphan SI (ISS-008)
- `tab:training_hyperparameters` | edit | Gamma italic (ISS-004)
- `tab:encoding_ablation_*` | keep | Headers OK

## Do-not-change
- Methodology subsection order (Design → FEA → FNO → Wavelets → Dataset → Training).
- Methodology focus on wavelet / 3-channel FNO I/O (aside from clarifying what the geometry channel is—ISS-001).
- SI Algorithms for Input Wavelet Encoding (wavelet-only).
- Encoding formulas for constant and sinusoidal in Results.
- Softened uniform-encoding language (fragile / high LR / tables for best case).
- Similarity math structure (\(\psi_{\mathbf{k}}\) 2D, \(\widehat{\psi}_{\mathbf{k}}\) flattened FFT of 2D field, cosine \(S\)).
- Material table numerical values \(E=10^8,\,2\times10^{11}\), \(\rho=1200,8000\), \(\nu=0.45,0.3\).
- Conclusions emphasis on design-loop acceleration and future work on symmetries / multimaterial spaces.
- NMAE as selected training objective; batch 520; step size 1; WD 0.
- Field-gallery figure count / layout (ISS-009 closed as nonissue).
- Minimize text changes to accomplish fixes.
