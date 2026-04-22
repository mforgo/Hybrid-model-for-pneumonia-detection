# Enhancement Documentation Guide

> Mapping ENHANCEMENTS_GUIDE.md improvements to soc.tex thesis sections  
> Target: Comprehensive hybrid QML paper for SOČ 2026

---

## Overview

This document details which enhancements from the SOTA pipeline should be integrated into the thesis (`soc.tex`) and where. Each enhancement includes theoretical justification, proposed text, and citation references.

---

## Enhancement Summary Table

| # | Enhancement | ENHANCEMENTS_GUIDE Reference | soc.tex Section | New Lines |
|---|------------|---------------------------|----------------|----------|
| 1 | VAE (Variational Autoencoder) | §1 (Autoencoder) | 2.4.4 | ~40 |
| 2 | Enhanced Encoding | §1 (Encoding) | 3.1.3 | ~20 |
| 3 | Trainable Measurement | §1 (Measurement) | 3.1.3 | ~15 |
| 4 | Linear Entanglement | §1 (Entanglement) | 3.1.2 | ~10 |
| 5 | QNG Optimizer | §1 (Optimizer) | 3.2.1 | ~25 |
| 6 | ZNE Mitigation | §1 (Error Mitigation) | 5.4 | ~35 |
| 7 | Parameter Count Analysis | Parameter Table | 3.3 | ~15 |

---

## Detail: Enhancement 1 - Variational Autoencoder (VAE)

### Location in soc.tex
**New Section 2.4.4**, after existing Section 2.4.3 (Autoencoders)

### Current State
- soc.tex currently describes MSE-only autoencoder (deterministic)
- No probabilistic latent space

### Enhancement Description

Replace deterministic MSE autoencoder with **Variational Autoencoder (VAE)** using reparameterization trick and KL divergence regularization.

### Proposed Theoretical Addition

```latex
\subsection{Variational Autoencoder (VAE)} \label{sec:vae}

Místo deterministického autoenkodéru jsme v experimentální části použili
variationalní autoenkodér (VAE), který modeluje latentní prostor jako
pravděpodobnostní distribuci místo bodového odhadu \cite{kingma_vae}.

Reprezentace $z$ je generována pomocí střední hodnoty $\mu$ a rozptylu $\sigma^2$:
\begin{equation}
    q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))^2
\end{equation}
\myequation{Parametrizace latentní distribuce pomocí sítě enkodéru.}

Kvůli zpětnému šíření používáme reparameterizační trik:
\begin{equation}
    z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\end{equation}
\myequation{Reparameterizační trik pro diferenciovatelný vzork.}

Celková ztráta VAE:
\begin{equation}
    \mathcal{L}_{VAE} = \mathcal{L}_{rec} + \beta \cdot D_{KL}(q(z|x) \| \mathcal{N}(0, I))
\end{equation}
\myequation{VAE ztráta: rekonstrukce + KL regularizace s faktorem $\beta$.}
```

### Key Citations
- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR 2014.

### Parameter Change
| Component | Old | New |
|-----------|-----|-----|
| Latent distribution | Point estimate $\mathbf{z}$ | $\mathcal{N}(\mu, \sigma^2)$ |
| KL term | None | $\beta \cdot D_{KL}$ |
| $\beta$ parameter | — | 1.0 |

---

## Detail: Enhancement 2 - Enhanced Encoding

### Location in soc.tex
**New Section 3.1.3**, after data re-uploading (Section 3.1.2)

### Current State
- Fixed encoding: AmplitudeEmbedding with $||\mathbf{x}||_2 = 1$
- No learnable scale parameter

### Enhancement Description

Add learnable encoding scale $w$ to allow dynamic range adaptation:

```latex
\subsection{Enhanced Encoding with Learnable Scale} \label{sec:enhanced_encoding}

Pro lepší přizpůsobení rozsahu vstupních dat jsme rozšířili základní
amplitude encoding o naučitelný parametr měřítka $w$. Namísto pevného
kódování $R_y(x_i \cdot \pi)$ používáme:

\begin{equation}
    R_y(w_i \cdot x_i \cdot \pi)
\end{equation}
\myequation{Amplitude encoding s naučitelným faktorem měřítka $w_i$.}

Parametr $w \in \mathbb{R}^n$ je inicializován na $1.0$ a trénován spolu
s parametry obvodu, což umožňuje modelu automaticky přizpůsobit dinamický
rozsah vstupních příznaků bez manuálního škálování.
```

### Key Citations
- Pérez-Salinas et al. (2020). *Quantum* 4:226.

### Parameter Count
- Added parameters: 6 (one scale per qubit)

---

## Detail: Enhancement 3 - Trainable Measurement Basis

### Location in soc.tex
**Section 3.1.3** (combined with Enhanced Encoding)

### Current State
- Fixed measurement on qubit 0: `qml.expval(qml.PauliZ(0))`

### Enhancement Description

Instead of fixed Pauli-Z measurement, use trainable rotation to basis:

```latex
\subsection{Trainable Měřící Báze} \label{sec:trainable_measurement}

Pro maximalizaci diskriminační schopnosti jsme nahradili fixní měření
Pauli-Z rotací trainable bází:

\begin{equation}
    R_y(\theta_{ry}) \rightarrow R_z(\theta_{rz}) \rightarrow \text{PauliZ}
\end{equation}
\myequation{Parametrizovaná měřící báze: rotace před měřením.}

Toto rozšíření umožňuje modelu naučit se optimální úhel pro separaci
tříd v Hilbertově prostoru, místo aby byl omezen na pevnou osu z.
```

### Parameter Count
- Added parameters: 2 (θ_ry, θ_rz)

---

## Detail: Enhancement 4 - Linear Entanglement

### Location in soc.tex
**Section 3.1.2** (modify existing ring CNOT)

### Current State
- Ring entanglement: `CNOT(wires=[w, (w+1) % n])`
- Requires SWAP on linear hardware topology

### Enhancement Modification

```latex
% In Section 3.1.2, modify the CNOT pattern:

Namísto kruhového (ring) propojení jsme použili lineární strukturu:
\begin{equation}
    \text{CNOT}(i, i+1) \quad \text{pro } i = 0, \dots, n-2
\end{equation}
\myequation{Lineární CNOT propojení bez SWAP režie.}
```

### Key Citations
- IBM Quantum documentation on hardware topology.

### Hardware Advantage
| Pattern | Ring | Linear |
|---------|------|--------|
| SWAPs needed | 0 | 0 |
| IBM T4 compatible | ✗ | ✓ |
| IBM Brisbane | ✗ | ✓ |

---

## Detail: Enhancement 5 - QNG Optimizer

### Location in soc.tex
**New Section 3.2.1**, after optimizer description

### Current State
- Custom Adam optimizer

### Enhancement Description

```latex
\subsection{Quantum Natural Gradient (QNG)} \label{sec:qng}

Pro rychlejší konvergenci jsme experimentovali s Quantum Natural
Gradient (QNG) optimalizátorem, který využívá Fubini-Studyovu
metriku místo standardní Euclidean \cite{mcclean_qng}.

Standardní gradient sleduje:
\begin{equation}
    \theta_{k+1} = \theta_k - \eta \cdot \nabla \mathcal{L}
\end{equation}

QNG používá:
\begin{equation}
    \theta_{k+1} = \theta_k - \eta \cdot F^{-1} \nabla \mathcal{L}
\end{equation}
\myequation{Quantum natural gradient s maticí F (Fubini-Study).}

kde $F$ je Fubini-Studyova metrika, která zohledňuje geometrii
kvantového parametrického prostoru.
```

### Key Citations
- McClean, J. R. et al. (2015). *Barren plateaus in quantum neural network training landscapes.* arXiv:1803.11173.

### Note for Thesis
- Results may show faster convergence, but standard Adam is acceptable baseline

---

## Detail: Enhancement 6 - ZNE Error Mitigation

### Location in soc.tex
**New Section 5.4** (in results/methodology chapter)

### Current State
- No hardware evaluation in current soc.tex
- Future work only

### Enhancement Description

```latex
\subsection{ZNE Error Mitigation} \label{sec:zne}

Pro evaluaci na reálném kvantovém hardwaru IBM jsme aplikovali
Zero-Noise Extrapolation (ZNE) \cite{temme_zne}.

Základní myšlenkou je škálování hloubky obvodu pomocí gate folding:
\begin{equation}
    G \rightarrow G \cdot \lambda
\end{equation}
\myequation{Škálování noise faktorem $\lambda$.}

Richardsonova extrapolace:
\begin{equation}
    f(0) = \frac{\lambda_2 f(\lambda_1) - \lambda_1 f(\lambda_2)}{\lambda_2 - \lambda_1}
\end{equation}
\myequation{Richardsonova extrapolace k zero-noise limitu.}

Evaluace probíhala na 50 testovacích vzorcích při škálovacích
faktorech $\lambda \in \{1, 2, 3\}$.
```

### Key Citations
- Temme, K. et al. (2017). *PRL* 119:180509.

---

## Detail: Enhancement 7 - Parameter Count Update

### Location in soc.tex
**Section 3.3** (hyperparameter table)

### Updated Parameter Table

| Model | Component | Parameters | Change |
|-------|-----------|------------|--------|
| **VQC (old)** | Rot layers | 54 | baseline |
| **VQC (new)** | Rot layers | 54 | — |
| | Encoding scale | +6 | w ∈ ℝⁿ |
| | Measurement | +2 | θ_ry, θ_rz |
| | **Total** | **62** | +8 |
| MLP baseline | Linear(64→32→1) | ~2,113 | — |

### Update to Results Table

| Metric | MLP (classical) | VQC (quantum, old) | VQC (quantum, new) |
|--------|----------------|-------------------|-------------------|
| Accuracy | XX.X% | XX.X% | XX.X% |
| Trainable params | 2,113 | 54 | 62 |
| **Improvement** | — | 39× fewer | 34× fewer |

---

## Summary: Sections to Add/Modify

### New Sections to Create
1. **Section 2.4.4**: VAE theory (~40 lines)
2. **Section 3.1.3**: Enhanced encoding + trainable measurement (~35 lines)
3. **Section 3.2.1**: QNG optimizer (~25 lines)
4. **Section 5.4**: ZNE methodology (~35 lines)

### Existing Sections to Modify
1. **Section 3.1.2**: Change Ring → Linear CNOT
2. **Section 3.3**: Update parameter table
3. **Section 5.x**: Add new results columns

### Total New Content
- **New lines**: ~135
- **Modified lines**: ~30

---

## Citation List Additions

| Enhancement | Citation |
|------------|----------|
| VAE | Kingma & Welling, *ICLR* 2014 |
| QNG | McClean et al., *arXiv* 2015 |
| ZNE | Temme et al., *PRL* 2017 |
| Linear topology | IBM Quantum documentation |

---

## Verification Checklist

After integration, verify:

- [ ] VAE: KL divergence from N(0,I) ≈ 0 (β=1.0)
- [ ] VAE: Reconstruction MSE ≤ old MSE AE
- [ ] VQC: Parameter count = 62 (not 54)
- [ ] VQC: Linear CNOT pattern (no SWAP)
- [ ] QNG: Gradient norm ≥ 0.01 at epoch 1
- [ ] ZNE: Hardware results in Table 5.2
- [ ] Expressibility: L=3 remains optimal
- [ ] All citations present in literatura.bib