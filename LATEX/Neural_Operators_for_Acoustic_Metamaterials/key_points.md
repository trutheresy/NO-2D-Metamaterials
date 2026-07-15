# Key findings — Neural Operators for Acoustic Metamaterials

Working list of major scientific points to carry through the paper (Results, abstract, conclusions).

## Core findings (primary)

1. **Wavelet encodings exploit both FNO branches.**  
   Wavelet encodings make effective use of both the spectral and spatial branches of the FNO architecture and allow the FNO to learn more complicated PDE mappings. Constant-field encodings starve the Fourier branch (only DC survives); sinusoidal encodings help but remain spectrally sparse and tend to produce fuzzy / under-resolved fields. Wavelets keep information rich in *both* domains so both branches can train and contribute.

2. **FNOs can learn eigenvalue PDEs with appropriate parameter encodings.**  
   With band and wavevector encodings that uniquely identify each eigenmode, an FNO can solve eigenvalue / multipartite-solution PDEs: multiple valid displacement modes for the same geometry, selected deterministically by the conditioning inputs. To our knowledge this is among the first demonstrations of FNOs on eigenvalue PDEs of this kind.

3. **Continuous inputs outperform discrete (binary) inputs.**  
   Continuous-valued geometries perform better than discrete-valued ones because discontinuities are broadband phenomena in Fourier space and are therefore poorly represented by a truncated, finite-mode FNO. Sharp 0/1 interfaces inject high-wavenumber content that the retained modes do not fully resolve.

4. **Mode toggling is deterministic and user-controlled.**  
   For a fixed geometry, wavevector and band inputs select among deformation modes on demand, without retraining or separate models per band / \(k\).

5. **One model spans continuous and discrete geometry distributions.**  
   The same wavelet-conditioned FNO learns both continuous-valued and binarized unit cells at usable accuracy (most test NMAE in ~\(10^{-2}\)–\(10^{-4}\)). That suggests some learning of the PDE operator itself beyond a single geometry prior—worth emphasizing carefully and flagging for future work.

6. **Among discrete designs, error tracks interface complexity and band index.**  
   Displacement MAE rises with boundary / interface length (Spearman \(\rho \approx 0.42\)–\(0.64\) by band) and is systematically larger for higher eigenbands. Harder cases are those with more extensive discontinuities and more oscillatory modes—consistent with the spectral-truncation story in (3).

## Additional major points from the draft (proposed)

7. **Encoding must match architecture transforms.**  
   For networks with built-in spatial–spectral structure (FNO), embeddings should preserve information in both domains. A purely spatial or purely reciprocal encoding of the same PDE parameters does not yield the same eigenvalue-solving capability. (The draft even notes that an FNO is in this sense closer to a “wavelet neural operator.”)

8. **Eigenfrequencies (dispersion) are recovered accurately as a byproduct.**  
   Encoded eigenfrequency / dispersion reconstruction on unseen continuous and discrete geometries averages $<\!1\%$ relative error on the test set (with the usual caveat that log-scale encoding makes absolute deviations look larger on higher bands in linear plots).

9. **Surrogate practicality.**  
   The model is small (~2 GB at float16) and evaluates in milliseconds, vs. expensive repeated FEA eigenvalue solves—relevant for iterative design, UQ, and optimization.

10. **Training / representation practicalities (secondary, but paper-supported).**  
    - Float16 real/imag channels match complex64 accuracy within ~10% relative difference but train \(1.5\)–\(2\times\) faster → prefer float16 unless precision is critical.  
    - Huber loss outperformed MAE / MSE / normalized variants / SSI among criteria tried.  
    - Learning rates \(\gtrsim 10^{-2}\) induce Adam-family loss spikes that do not recover; step size and batch size had little effect once fixed.  
    - Model capacity asymptotes around a modest FNO (draft discusses ~4 layers / 64–128 channels + GELU); more capacity mainly adds cost / overfitting risk.

## Gaps / unfinished storylines in the draft

- Encoding ablation (constant → sinusoid → wavelet) is only partly written; sinusoid/wavelet figure blocks are still commented out.  
- Dedicated Results write-up of hyperparameters and loss is still missing relative to the Results checklist.  
- Claims of “first” application and related work citations still need hardening.
