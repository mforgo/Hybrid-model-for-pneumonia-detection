"""Variational quantum circuit builder and expressibility analysis.

Task 8 of the notebook → package rewrite. Implements the data re-uploading
VQC ansatz used by the hybrid quantum-classical pneumonia pipeline:

* ``build_vqc_circuit`` — constructs the trainable ``qml.QNode``.
* ``count_params``     — pure arithmetic parameter counter (62 default / 54
  when both ablation flags are off). This is the acceptance contract for
  ``tests/test_ansatz.py``.
* ``expressibility_sweep`` / ``entanglement_capability`` — Sim et al. (2019)
  expressibility (KL divergence vs. the Haar measure) and Meyer-Wallach
  entanglement capability of the ansatz.

Design decisions (LOCKED by the rewrite plan):

* Circuit: ``AmplitudeEmbedding`` once, then ``n_layers`` blocks of
  [RY data re-upload, ``Rot``, ring CNOT], optional trainable measurement
  basis, ``expval(PauliZ(0))``.
* Trainable parameters are packed into a *single flat tensor* of shape
  ``(count_params(...),)`` with layout::

      [rot (n_layers*n_qubits*3), scale (n_qubits,), meas (2,)]

* Differentiation: ``adjoint`` on lightning devices, ``parameter-shift`` on
  ``default.qubit``. ``adjoint`` is NEVER used on ``default.qubit``.

All heavy imports (``pennylane``, ``numpy``) are performed lazily inside the
functions so the module imports with the standard library alone.
"""

from __future__ import annotations

SEED = 6  # project-wide random seed (AGENTS.md global rule)


def count_params(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_scale: bool = True,
    use_meas_basis: bool = True,
) -> int:
    """Number of trainable parameters of the VQC ansatz.

    Formula: ``3 * n_layers * n_qubits`` (Euler angles of the ``Rot`` gates)
    plus ``n_qubits`` when the learnable input scale is enabled plus ``2``
    when the trainable measurement basis is enabled.

    Pure arithmetic — no PennyLane required, so it is verifiable in any
    environment.

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of data re-uploading layers (default 3).
        use_scale: Include the ``n_qubits`` learnable input scale parameters.
        use_meas_basis: Include the 2 trainable measurement-basis parameters.

    Returns:
        Total number of trainable parameters. 62 for the default flags,
        54 when both ablation flags are ``False``.
    """
    n_rot = n_layers * n_qubits * 3
    n_scale = n_qubits if use_scale else 0
    n_meas = 2 if use_meas_basis else 0
    return n_rot + n_scale + n_meas


def split_params(
    params,
    n_qubits: int = 6,
    n_layers: int = 3,
    use_scale: bool = True,
    use_meas_basis: bool = True,
):
    """Split a flat ``params`` tensor into ``(rot, scale, meas)`` components.

    Layout (mirrors ``count_params`` ordering)::

        [rot (n_layers*n_qubits*3), scale (n_qubits,), meas (2,)]

    ``rot`` is reshaped to ``(n_layers, n_qubits, 3)``. ``scale`` / ``meas``
    are ``None`` when the corresponding flag is off.

    Pure NumPy — works without PennyLane (useful for the VQC training loop
    and for QPU inference, which only needs the ``rot`` block).
    """
    import numpy as np

    n_rot = n_layers * n_qubits * 3
    n_scale = n_qubits if use_scale else 0
    n_meas = 2 if use_meas_basis else 0

    rot = np.asarray(params[:n_rot]).reshape((n_layers, n_qubits, 3))
    scale = np.asarray(params[n_rot : n_rot + n_scale]) if use_scale else None
    meas = np.asarray(params[n_rot + n_scale :]) if use_meas_basis else None
    return rot, scale, meas


def _is_lightning_device(dev) -> bool:
    """True when ``dev`` is a PennyLane lightning device (supports adjoint)."""
    name = getattr(dev, "short_name", None) or getattr(dev, "name", "") or ""
    return "lightning" in str(name)


def _resolve_device(n_qubits: int, device=None, diff_method: str | None = None):
    """Pick the best available PennyLane device and a matching diff method.

    Fallback chain (LOCKED): ``lightning.gpu`` → ``lightning.qubit`` →
    ``default.qubit``. ``adjoint`` is only used on lightning devices;
    ``default.qubit`` always uses ``parameter-shift`` (adjoint is never used
    there, even if explicitly requested).

    Args:
        n_qubits: Number of wires for auto-created devices.
        device: ``None`` (auto-resolve), a device name string, or an existing
            ``qml.Device`` instance.
        diff_method: Requested differentiation method. ``None`` selects
            ``"adjoint"`` on lightning devices and ``"parameter-shift"`` on
            ``default.qubit``.

    Returns:
        ``(dev, diff_method)`` tuple.
    """
    import pennylane as qml

    if device is None:
        for name in ("lightning.gpu", "lightning.qubit"):
            try:
                return qml.device(name, wires=n_qubits), "adjoint"
            except Exception:
                continue
        return qml.device("default.qubit", wires=n_qubits), "parameter-shift"

    if isinstance(device, str):
        dev = qml.device(device, wires=n_qubits)
    else:
        dev = device

    if _is_lightning_device(dev):
        return dev, diff_method or "adjoint"
    # default.qubit (or any non-lightning device): NEVER adjoint.
    return dev, "parameter-shift"


def build_vqc_circuit(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_scale: bool = True,
    use_meas_basis: bool = True,
    device=None,
    diff_method: str | None = None,
):
    """Build the variational quantum classifier QNode.

    Circuit structure (LOCKED)::

        AmplitudeEmbedding(features[:2**n_qubits], normalize=True, pad_with=0.0)   # once
        for l in range(n_layers):
            for q in range(n_qubits):
                RY(scale[q] * features[q] * pi, q)      # data re-upload (scale only if use_scale)
            for q in range(n_qubits):
                Rot(params[l,q,0], params[l,q,1], params[l,q,2], q)
            for q in range(n_qubits):
                CNOT(q, (q + 1) % n_qubits)             # ring entangler
        if use_meas_basis:
            RY(meas[0], 0); RZ(meas[1], 0)
        return expval(PauliZ(0))

    Trainable parameters are passed as a single flat tensor of shape
    ``(count_params(...),)`` (see ``split_params`` for the layout). The
    second argument is the 64-dimensional (L2-normalised) feature vector.

    Diff method: ``adjoint`` on lightning devices, ``parameter-shift`` on
    ``default.qubit`` (never adjoint there).

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of data re-uploading layers (default 3).
        use_scale: Include the learnable input scale parameters.
        use_meas_basis: Include the trainable measurement basis.
        device: ``None`` (auto-resolve fallback chain), a device name string,
            or an existing ``qml.Device`` instance.
        diff_method: Requested differentiation method (``None`` = auto).

    Returns:
        A ``qml.QNode`` callable as ``circuit(params, features)`` returning
        ``⟨Z₀⟩ ∈ [-1, 1]``.
    """
    import numpy as np
    import pennylane as qml

    dev, diff_method = _resolve_device(n_qubits, device=device, diff_method=diff_method)

    n_rot = n_layers * n_qubits * 3
    n_scale = n_qubits if use_scale else 0
    n_meas = 2 if use_meas_basis else 0
    n_amp = 2**n_qubits

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params, features):
        # --- split the flat trainable tensor (layout matches count_params) ---
        rot = params[:n_rot].reshape((n_layers, n_qubits, 3))
        scale = params[n_rot : n_rot + n_scale] if use_scale else None
        meas = params[n_rot + n_scale :] if use_meas_basis else None

        # Encode the input ONCE via amplitude embedding (features are already
        # L2-normalised by the VAE/PCA pipeline; normalize=True is a safety net).
        qml.AmplitudeEmbedding(
            features[:n_amp], wires=range(n_qubits), normalize=True, pad_with=0.0
        )

        # Data re-uploading layers: RY re-upload + trainable Rot + ring CNOT.
        for l in range(n_layers):
            for q in range(n_qubits):
                if use_scale:
                    qml.RY(scale[q] * features[q] * np.pi, wires=q)
                else:
                    qml.RY(features[q] * np.pi, wires=q)
            for q in range(n_qubits):
                qml.Rot(rot[l, q, 0], rot[l, q, 1], rot[l, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        # Trainable measurement basis on qubit 0 (only if enabled).
        if use_meas_basis:
            qml.RY(meas[0], wires=0)
            qml.RZ(meas[1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return circuit


# ---------------------------------------------------------------------------
# Angle-encoding ansatz
# ---------------------------------------------------------------------------


def count_params_angle(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_scale: bool = True,
    use_meas_basis: bool = True,
) -> int:
    """Number of trainable parameters of the angle-encoding VQC ansatz.

    Formula: ``3 * n_layers * n_qubits`` (Euler angles of the ``Rot`` gates)
    plus ``n_layers * n_qubits`` (per-layer trainable RY re-upload angles)
    plus ``n_qubits`` when the learnable input scale is enabled plus ``2``
    when the trainable measurement basis is enabled.  80 for the default
    flags.

    Pure arithmetic — no PennyLane required.

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of data re-uploading layers (default 3).
        use_scale: Include the ``n_qubits`` learnable input scale parameters.
        use_meas_basis: Include the 2 trainable measurement-basis parameters.

    Returns:
        Total number of trainable parameters.
    """
    n_rot = n_layers * n_qubits * 3
    n_angle = n_layers * n_qubits
    n_scale = n_qubits if use_scale else 0
    n_meas = 2 if use_meas_basis else 0
    return n_rot + n_angle + n_scale + n_meas


def build_vqc_circuit_angle(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_scale: bool = True,
    use_meas_basis: bool = True,
    device=None,
    diff_method: str | None = None,
):
    """Build the angle-encoding VQC QNode.

    Circuit structure (LOCKED)::

        for l in range(n_layers):
            for q in range(n_qubits):
                RY(angle[l,q] + scale[q] * features[q] * pi, q)   # trainable angle re-upload
            for q in range(n_qubits):
                Rot(rot[l,q,0], rot[l,q,1], rot[l,q,2], q)
            for q in range(n_qubits):
                CNOT(q, (q + 1) % n_qubits)             # ring entangler
        if use_meas_basis:
            RY(meas[0], 0); RZ(meas[1], 0)
        return expval(PauliZ(0))

    Unlike :func:`build_vqc_circuit` (amplitude embedding), the input is
    encoded via per-layer RY rotations of the first ``n_qubits`` features.
    Each layer has its own trainable angle offset ``angle[l, q]``; the
    optional learnable scale multiplies the data contribution.

    Trainable parameters are passed as a single flat tensor of shape
    ``(count_params_angle(...),)`` with layout::

        [rot (n_layers*n_qubits*3), angle (n_layers*n_qubits),
         scale (n_qubits,), meas (2,)]

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of data re-uploading layers (default 3).
        use_scale: Include the learnable input scale parameters.
        use_meas_basis: Include the trainable measurement basis.
        device: ``None`` (auto-resolve), a device name string,
            or an existing ``qml.Device`` instance.
        diff_method: Requested differentiation method (``None`` = auto).

    Returns:
        A ``qml.QNode`` callable as ``circuit(params, features)`` returning
        ``⟨Z₀⟩ ∈ [-1, 1]``.
    """
    import numpy as np
    import pennylane as qml

    dev, diff_method = _resolve_device(n_qubits, device=device, diff_method=diff_method)

    n_rot = n_layers * n_qubits * 3
    n_angle = n_layers * n_qubits
    n_scale = n_qubits if use_scale else 0
    n_meas = 2 if use_meas_basis else 0

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params, features):
        rot = params[:n_rot].reshape((n_layers, n_qubits, 3))
        angle = params[n_rot : n_rot + n_angle].reshape((n_layers, n_qubits))
        scale = params[n_rot + n_angle : n_rot + n_angle + n_scale] if use_scale else None
        meas = params[n_rot + n_angle + n_scale :] if use_meas_basis else None

        for l in range(n_layers):
            for q in range(n_qubits):
                if use_scale:
                    qml.RY(angle[l, q] + scale[q] * features[q] * np.pi, wires=q)
                else:
                    qml.RY(angle[l, q] + features[q] * np.pi, wires=q)
            for q in range(n_qubits):
                qml.Rot(rot[l, q, 0], rot[l, q, 1], rot[l, q, 2], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        if use_meas_basis:
            qml.RY(meas[0], wires=0)
            qml.RZ(meas[1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return circuit


# ---------------------------------------------------------------------------
# Hardware-efficient ansatz
# ---------------------------------------------------------------------------


def count_params_he(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_meas_basis: bool = True,
) -> int:
    """Number of trainable parameters of the hardware-efficient ansatz.

    Formula: ``2 * n_layers * n_qubits`` (RY/RZ rotations per qubit per
    layer) plus ``2`` when the trainable measurement basis is enabled.
    38 for the default flags.

    Pure arithmetic — no PennyLane required.

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of ansatz layers (default 3).
        use_meas_basis: Include the 2 trainable measurement-basis parameters.

    Returns:
        Total number of trainable parameters.
    """
    n_rot = n_layers * n_qubits * 2
    n_meas = 2 if use_meas_basis else 0
    return n_rot + n_meas


def build_vqc_circuit_he(
    n_qubits: int = 6,
    n_layers: int = 3,
    use_meas_basis: bool = True,
    device=None,
    diff_method: str | None = None,
):
    """Build the hardware-efficient ansatz (HEA) VQC QNode.

    Circuit structure (LOCKED)::

        for q in range(n_qubits):
            RY(features[q] * pi, q)                     # angle embedding (data)
        for l in range(n_layers):
            for q in range(n_qubits):
                RY(rot[l,q,0], q); RZ(rot[l,q,1], q)    # trainable single-qubit rotations
            for q in range(n_qubits):
                CNOT(q, (q + 1) % n_qubits)             # ring entangler
        if use_meas_basis:
            RY(meas[0], 0); RZ(meas[1], 0)
        return expval(PauliZ(0))

    The input is angle-embedded once via ``RY(features[q] * pi)``; the
    trainable part is a hardware-efficient stack of single-qubit RY/RZ
    rotations interleaved with nearest-neighbour CNOT entanglers.

    Trainable parameters are passed as a single flat tensor of shape
    ``(count_params_he(...),)`` with layout::

        [rot (n_layers*n_qubits*2), meas (2,)]

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of ansatz layers (default 3).
        use_meas_basis: Include the trainable measurement basis.
        device: ``None`` (auto-resolve), a device name string,
            or an existing ``qml.Device`` instance.
        diff_method: Requested differentiation method (``None`` = auto).

    Returns:
        A ``qml.QNode`` callable as ``circuit(params, features)`` returning
        ``⟨Z₀⟩ ∈ [-1, 1]``.
    """
    import numpy as np
    import pennylane as qml

    dev, diff_method = _resolve_device(n_qubits, device=device, diff_method=diff_method)

    n_rot = n_layers * n_qubits * 2
    n_meas = 2 if use_meas_basis else 0

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params, features):
        rot = params[:n_rot].reshape((n_layers, n_qubits, 2))
        meas = params[n_rot:] if use_meas_basis else None

        for q in range(n_qubits):
            qml.RY(features[q] * np.pi, wires=q)

        for l in range(n_layers):
            for q in range(n_qubits):
                qml.RY(rot[l, q, 0], wires=q)
                qml.RZ(rot[l, q, 1], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])

        if use_meas_basis:
            qml.RY(meas[0], wires=0)
            qml.RZ(meas[1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return circuit


def _core_ansatz_circuit(n_qubits: int, n_layers: int):
    """Build the core ansatz QNode (rot params only) returning the full state.

    Used by the expressibility / entanglement analysis. Matches the original
    notebook's ``make_vqc_circuit``: amplitude embedding once, then L layers
    of [RY re-upload, Rot, ring CNOT]. No scale / measurement-basis params —
    the analysis characterises the ansatz unitary itself (Sim et al. 2019).
    """
    import numpy as np
    import pennylane as qml

    dev, _ = _resolve_device(n_qubits)
    n_amp = 2**n_qubits

    @qml.qnode(dev)
    def circuit(x, params):
        qml.AmplitudeEmbedding(
            x[:n_amp], wires=range(n_qubits), normalize=True, pad_with=0.0
        )
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(x[w] * np.pi, wires=w)
            for w in range(n_qubits):
                qml.Rot(params[l, w, 0], params[l, w, 1], params[l, w, 2], wires=w)
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])
        return qml.state()

    return circuit


def _compute_expressibility(
    n_qubits: int, n_layers: int, n_samples: int, n_bins: int, rng
) -> float:
    """KL divergence of the ansatz state-fidelity distribution from Haar.

    Samples ``n_samples`` pairs of random parameter sets, applies the ansatz
    to a fixed random input state, and compares the distribution of output
    state fidelities to the Haar-random distribution ``Beta(1, 2^n - 1)``.
    Lower values indicate greater expressibility.
    """
    import numpy as np

    dim = 2**n_qubits
    circuit = _core_ansatz_circuit(n_qubits, n_layers)

    fidelities = np.empty(n_samples)
    for i in range(n_samples):
        params_a = rng.uniform(0.0, 2.0 * np.pi, (n_layers, n_qubits, 3))
        params_b = rng.uniform(0.0, 2.0 * np.pi, (n_layers, n_qubits, 3))
        x = rng.normal(size=dim)
        x = x / np.linalg.norm(x)
        state_a = circuit(x, params_a)
        state_b = circuit(x, params_b)
        fidelities[i] = np.abs(np.vdot(state_a, state_b)) ** 2

    # Ansatz fidelity histogram (density-normalised).
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    p_ansatz = hist + 1e-12
    p_ansatz /= p_ansatz.sum()

    # Haar distribution: Beta(1, dim - 1)  ->  pdf(x) = (dim-1) * (1-x)^(dim-2).
    p_haar = (dim - 1.0) * (1.0 - bin_centers) ** (dim - 2)
    p_haar = p_haar + 1e-12
    p_haar /= p_haar.sum()

    # Standard KL divergence (equivalent to scipy.special.kl_div summed over
    # two normalised histograms, where the extra -P+Q terms telescope to 0).
    kl = np.sum(p_ansatz * np.log(p_ansatz / p_haar))
    return float(kl)


def _compute_entanglement_capability(
    n_qubits: int, n_layers: int, n_samples: int, rng
) -> float:
    """Meyer-Wallach entanglement measure averaged over random parameters.

    ``Q = (2/n) * sum_k (1 - Tr(rho_k^2))`` where ``rho_k`` is the reduced
    density matrix of qubit ``k``. Averaged over ``n_samples`` random
    parameter sets and random input states.
    """
    import numpy as np

    dim = 2**n_qubits
    circuit = _core_ansatz_circuit(n_qubits, n_layers)

    mw_measures = np.empty(n_samples)
    for i in range(n_samples):
        params = rng.uniform(0.0, 2.0 * np.pi, (n_layers, n_qubits, 3))
        x = rng.normal(size=dim)
        x = x / np.linalg.norm(x)
        state = circuit(x, params)

        rho = np.outer(state, np.conj(state))
        q = 0.0
        for k in range(n_qubits):
            # Trace out all qubits except k to get the single-qubit reduced
            # density matrix rho_k.
            remaining = [j for j in range(n_qubits) if j != k]
            rho_tensor = rho.reshape([2] * (2 * n_qubits))
            axes_bra = [k] + remaining
            axes_ket = [k + n_qubits] + [j + n_qubits for j in remaining]
            rho_tensor = np.transpose(rho_tensor, axes_bra + axes_ket)
            rho_tensor = rho_tensor.reshape([2, 2 ** (n_qubits - 1), 2, 2 ** (n_qubits - 1)])
            rho_k = np.trace(rho_tensor, axis1=1, axis2=3)
            purity = np.real(np.trace(rho_k @ rho_k))
            q += 1.0 - purity
        q *= 2.0 / n_qubits
        mw_measures[i] = q

    return float(np.mean(mw_measures))


def expressibility_sweep(
    n_qubits: int = 6,
    layers=(1, 2, 3, 4),
    n_samples: int = 2000,
    n_bins: int = 75,
    seed: int = SEED,
):
    """Sweep expressibility and entanglement capability over layer counts.

    For each ``L`` in ``layers`` computes the Sim et al. (2019) expressibility
    (KL divergence from the Haar measure) and the Meyer-Wallach entanglement
    capability, plus the parameter count ``L * n_qubits * 3``.

    Args:
        n_qubits: Number of qubits (default 6).
        layers: Iterable of layer counts to sweep (default ``(1, 2, 3, 4)``).
        n_samples: Number of random parameter sets per layer count.
        n_bins: Number of histogram bins for the fidelity distribution.
        seed: Random seed (project convention 6).

    Returns:
        List of dicts with keys ``n_layers``, ``Expr(A)``, ``Ent(A)``,
        ``n_params`` — ready for ``pandas.DataFrame`` / CSV export.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    results = []
    for n_layers in layers:
        expr = _compute_expressibility(n_qubits, n_layers, n_samples, n_bins, rng)
        ent = _compute_entanglement_capability(n_qubits, n_layers, n_samples, rng)
        results.append(
            {
                "n_layers": n_layers,
                "Expr(A)": round(expr, 4),
                "Ent(A)": round(ent, 4),
                "n_params": n_layers * n_qubits * 3,
            }
        )
    return results


def entanglement_capability(
    n_qubits: int = 6,
    n_layers: int = 3,
    n_samples: int = 2000,
    seed: int = SEED,
) -> float:
    """Meyer-Wallach entanglement measure averaged over random parameters.

    Args:
        n_qubits: Number of qubits (default 6).
        n_layers: Number of ansatz layers (default 3).
        n_samples: Number of random parameter sets to average over.
        seed: Random seed (project convention 6).

    Returns:
        Mean Meyer-Wallach entanglement ``Q ∈ [0, 1]``.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return _compute_entanglement_capability(n_qubits, n_layers, n_samples, rng)


__all__ = [
    "SEED",
    "build_vqc_circuit",
    "build_vqc_circuit_angle",
    "build_vqc_circuit_he",
    "count_params",
    "count_params_angle",
    "count_params_he",
    "split_params",
    "expressibility_sweep",
    "entanglement_capability",
]