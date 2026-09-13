"""IBM Quantum hardware evaluation: token discovery, QPU inference, ZNE, and
the FakeKingston noisy-simulator baseline.

Task 12 of the notebook → package rewrite. Consumes the trained VQC from
``pipeline.vqc`` (``results/vqc_best_params.npy`` + ``split_params`` layout
``[rot (n_layers*n_qubits*3), scale (n_qubits,), meas (2,)]``) and the
``qml.QNode`` built by ``pipeline.ansatz.build_vqc_circuit``.

Design decisions (LOCKED by the rewrite plan):

* Token precedence: env ``IBM_TOKEN`` > ``ibm_token.txt`` (repo root) >
  ``cfg.ibm_token``.
* Hardware access is **job mode only** — never ``Session`` (Session bills
  wall-clock time).
* Qiskit bitstrings are **big-endian**: qubit 0 is the LAST character of the
  bitstring (bug fix #3).
* All heavy imports (qiskit, qiskit-ibm-runtime, qiskit-aer, mitiq,
  pennylane-qiskit) are lazy *inside* the functions so this module imports
  with the standard library + NumPy alone. Every hardware path degrades
  gracefully (warn + ``None``) when the stack is missing or the backend is
  unreachable.

Artifacts (per plan Data Flow)::

    results/vqc_qpu_probs.npy          QPU inference probabilities
    results/vqc_qpu_sim_probs.npy      ideal-simulator probabilities
    results/vqc_fakekingston_probs.npy FakeKingston noisy-sim probabilities
    results/zne_comparison.csv         sim vs noisy-sim vs raw-QPU vs ZNE table
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import numpy as np

SEED = 6  # project-wide random seed (AGENTS.md global rule)


def resolve_ibm_token(cfg) -> str | None:
    """Resolve the IBM Quantum API token with fixed precedence.

    Precedence (LOCKED): environment variable ``IBM_TOKEN`` >
    ``ibm_token.txt`` at the repo root (read if it exists) > ``cfg.ibm_token``.

    Args:
        cfg: The pipeline ``Config`` instance.

    Returns:
        The token string, or ``None`` when no source provides one.
    """
    env_token = os.environ.get("IBM_TOKEN", "")
    if env_token:
        return env_token
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        token_file = base / "ibm_token.txt"
        if token_file.is_file():
            token = token_file.read_text().strip()
            if token:
                return token
    if getattr(cfg, "ibm_token", ""):
        return cfg.ibm_token
    return None


def get_ibm_backend(cfg) -> object | None:
    """Connect to IBM Quantum and return a least-busy backend, or ``None``.

    Uses ``qiskit_ibm_runtime.QiskitRuntimeService``. When ``cfg.ibm_backend``
    is set it is used directly; otherwise ``service.least_busy(...)`` selects
    the least-busy operational real backend with at least ``cfg.n_qubits``
    qubits (the transpiler later picks 6 connected low-error qubits from the
    device calibration map).

    Args:
        cfg: The pipeline ``Config`` instance.

    Returns:
        A Qiskit ``BackendV2``, or ``None`` when no token is available, the
        qiskit stack is missing, or the connection fails (offline / queue /
        auth). A warning is emitted in every failure case.
    """
    token = resolve_ibm_token(cfg)
    if not token:
        if cfg.run_mode != "sim":
            warnings.warn(
                "No IBM Quantum token found (checked env IBM_TOKEN, "
                "ibm_token.txt, cfg.ibm_token); falling back to "
                "simulator-only evaluation.",
                RuntimeWarning,
                stacklevel=2,
            )
        return None

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        warnings.warn(
            "qiskit-ibm-runtime is not installed; cannot access IBM Quantum.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        service = QiskitRuntimeService(
            token=token,
            instance=cfg.ibm_instance or None,
        )
        if cfg.ibm_backend:
            return service.backend(cfg.ibm_backend)
        return service.least_busy(
            operational=True, simulator=False, min_num_qubits=cfg.n_qubits
        )
    except Exception as exc:  # offline / queue / auth — never hard-crash
        warnings.warn(
            f"Failed to connect to IBM Quantum: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def extract_z0_from_counts(counts, n_qubits) -> float:
    """Compute the Pauli-Z expectation value of qubit 0 from shot counts.

    Qiskit bitstrings are **big-endian**: the LAST character of the bitstring
    is the measurement of qubit 0 (bug fix #3). ``counts`` maps bitstrings of
    length ``n_qubits`` to shot counts.

    Args:
        counts: Dict ``{bitstring: shots}`` in Qiskit bit order.
        n_qubits: Number of qubits (bitstring length).

    Returns:
        ``⟨Z₀⟩ = (n_qubit0_measured_0 - n_qubit0_measured_1) / total_shots``
        in ``[-1, 1]``. Returns ``0.0`` for empty counts.
    """
    total = 0
    n_zero = 0
    for bitstring, shots in counts.items():
        total += int(shots)
        if bitstring[-1] == "0":
            n_zero += int(shots)
    if total == 0:
        return 0.0
    return (2.0 * n_zero - total) / total


def _get_to_qiskit():
    """Return a ``to_qiskit`` converter callable, or ``None`` if unavailable.

    Tries ``pennylane_qiskit.to_qiskit`` first (the plugin), then
    ``pennylane.qml.to_qiskit`` (re-exported in newer PennyLane).
    """
    try:
        from pennylane_qiskit import to_qiskit

        return to_qiskit
    except ImportError:
        pass
    try:
        import pennylane as qml

        return qml.to_qiskit
    except (ImportError, AttributeError):
        return None


def _qnode_to_qiskit(circuit_qnode, params, features):
    """Convert a bound ``qml.QNode`` to a Qiskit ``QuantumCircuit``.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        features: 64-dimensional L2-normalised feature vector.

    Returns:
        A Qiskit ``QuantumCircuit`` with all parameters bound.

    Raises:
        RuntimeError: If the pennylane-qiskit converter is unavailable.
    """
    to_qiskit = _get_to_qiskit()
    if to_qiskit is None:
        raise RuntimeError(
            "pennylane-qiskit is not installed; cannot convert the QNode to a "
            "Qiskit circuit. Install with: pip install pennylane-qiskit"
        )
    params = np.asarray(params, dtype=float)
    features = np.asarray(features, dtype=float)
    return to_qiskit(circuit_qnode)(params, features)


def _extract_counts(pub_result):
    """Extract a ``{bitstring: shots}`` dict from a SamplerV2 ``PubResult``.

    The measurement field name depends on the classical register created by
    the pennylane-qiskit converter, so the ``DataBin`` is scanned for the
    first field exposing ``get_counts()``.
    """
    data = pub_result.data
    for name in ("meas", "measurement", "c", "out"):
        try:
            field = getattr(data, name)
        except AttributeError:
            continue
        if hasattr(field, "get_counts"):
            return dict(field.get_counts())
    for name in dir(data):
        if name.startswith("_"):
            continue
        try:
            field = getattr(data, name)
        except AttributeError:
            continue
        if hasattr(field, "get_counts"):
            return dict(field.get_counts())
    raise RuntimeError("No measurement counts found in SamplerV2 result")


def make_ibm_executor(backend, shots=1024):
    """Build a mitiq-compatible executor: Qiskit circuit -> float ⟨Z₀⟩.

    The returned callable transpiles the circuit for *backend*
    (``optimization_level=3``), submits it via ``SamplerV2`` in job mode (no
    Session), and reduces the shot counts to a single ``⟨Z₀⟩`` value.

    Args:
        backend: A Qiskit ``BackendV2`` (e.g. from ``get_ibm_backend``).
        shots: Number of shots per circuit.

    Returns:
        ``executor(circuit) -> float``.
    """

    def executor(circuit):
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2

        tqc = transpile(circuit, backend=backend, optimization_level=3)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        counts = _extract_counts(job.result()[0])
        return extract_z0_from_counts(counts, tqc.num_qubits)

    return executor


def run_vqc_on_qpu(circuit_qnode, params, X_subset, backend, shots=1024) -> np.ndarray | None:
    """Run the trained VQC on real IBM hardware for a subset of samples.

    For each sample: convert the bound QNode to a Qiskit circuit, transpile
    with ``optimization_level=3``, submit via ``SamplerV2`` in job mode (no
    Session), extract shot counts, compute ``⟨Z₀⟩`` and map to probability
    ``(1 + ⟨Z₀⟩) / 2``.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        X_subset: Array of feature vectors, shape ``(n_samples, 64)``.
        backend: A Qiskit ``BackendV2`` (e.g. from ``get_ibm_backend``).
        shots: Number of shots per circuit (default 1024).

    Returns:
        Array of pneumonia probabilities, shape ``(n_samples,)``, or ``None``
        if the qiskit stack is unavailable (graceful degradation).
    """
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2
    except ImportError:
        warnings.warn(
            "qiskit / qiskit-ibm-runtime not installed; cannot run on QPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    to_qiskit = _get_to_qiskit()
    if to_qiskit is None:
        warnings.warn(
            "pennylane-qiskit not installed; cannot convert the QNode to Qiskit.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    params = np.asarray(params, dtype=float)
    X = np.asarray(X_subset, dtype=float)
    probs = np.empty(len(X), dtype=float)
    for i, x in enumerate(X):
        qc = to_qiskit(circuit_qnode)(params, x)
        tqc = transpile(qc, backend=backend, optimization_level=3)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([tqc], shots=shots)
        counts = _extract_counts(job.result()[0])
        z0 = extract_z0_from_counts(counts, tqc.num_qubits)
        probs[i] = (1.0 + z0) / 2.0
    return probs


def run_zne_mitigation(
    circuit_qnode, params, x_sample, backend, executor, scale_factors=(1, 2, 3)
) -> float:
    """Apply Zero-Noise Extrapolation to a single sample's expectation value.

    Uses ``mitiq.zne.execute_with_zne`` with a ``RichardsonFactory`` over the
    given noise scale factors. The *executor* wraps the transpile + SamplerV2
    + ``extract_z0_from_counts`` flow and returns a float ``⟨Z₀⟩``.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        x_sample: Single 64-dimensional feature vector.
        backend: A Qiskit ``BackendV2`` (used only when *executor* is ``None``).
        executor: Callable ``circuit -> float ⟨Z₀⟩`` (e.g. from
            ``make_ibm_executor``). When ``None``, one is built from *backend*.
        scale_factors: Noise scale factors for Richardson extrapolation.

    Returns:
        The ZNE-mitigated ``⟨Z₀⟩``. If mitiq is missing, returns the raw
        noisy value from *executor* with a warning.
    """
    if executor is None:
        executor = make_ibm_executor(backend)

    qc = _qnode_to_qiskit(circuit_qnode, params, x_sample)

    try:
        import mitiq
        from mitiq.zne.inference import RichardsonFactory
    except ImportError:
        warnings.warn(
            "mitiq not installed; returning raw noisy <Z0> without ZNE.",
            RuntimeWarning,
            stacklevel=2,
        )
        return executor(qc)

    factory = RichardsonFactory(scale_factors=list(scale_factors))
    return mitiq.zne.execute_with_zne(circuit=qc, executor=executor, factory=factory)


def run_fakekingston_baseline(
    circuit_qnode, params, X_subset, shots=1024
) -> np.ndarray | None:
    """Noisy-simulator baseline using AerSimulator + FakeKingston noise.

    Runs the VQC through ``qiskit_aer.AerSimulator`` with a noise model built
    from ``FakeKingston`` (a fake Heron r2 backend). Produces the noise
    degradation reference without consuming IBM queue time.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        X_subset: Array of feature vectors, shape ``(n_samples, 64)``.
        shots: Number of shots per circuit (default 1024).

    Returns:
        Array of noisy-simulator probabilities, shape ``(n_samples,)``, or
        ``None`` if qiskit-aer / FakeKingston is unavailable (warn + skip).
    """
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
    except ImportError:
        warnings.warn(
            "qiskit-aer not installed; skipping noisy baseline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # FakeKingston lives in different modules across qiskit-ibm-runtime
    # versions; try the modern location first, then the legacy one.
    try:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
    except ImportError:
        try:
            from qiskit.providers.fake_provider import FakeKingston
        except ImportError:
            warnings.warn(
                "FakeKingston unavailable; skipping noisy baseline.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    to_qiskit = _get_to_qiskit()
    if to_qiskit is None:
        warnings.warn(
            "pennylane-qiskit not installed; skipping noisy baseline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    noise_model = NoiseModel.from_backend(FakeKingston())
    backend = AerSimulator(noise_model=noise_model)

    params = np.asarray(params, dtype=float)
    X = np.asarray(X_subset, dtype=float)
    probs = np.empty(len(X), dtype=float)
    for i, x in enumerate(X):
        qc = to_qiskit(circuit_qnode)(params, x)
        tqc = transpile(qc, backend=backend, optimization_level=3)
        job = backend.run(tqc, shots=shots)
        counts = job.result().get_counts()
        z0 = extract_z0_from_counts(counts, tqc.num_qubits)
        probs[i] = (1.0 + z0) / 2.0
    return probs


def _save_artifacts(cfg, result) -> None:
    """Persist QPU artifacts per plan Data Flow.

    Writes ``results/vqc_qpu_probs.npy``, ``results/vqc_qpu_sim_probs.npy``,
    ``results/vqc_fakekingston_probs.npy`` and ``results/zne_comparison.csv``
    for every non-``None`` entry in *result*.
    """
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.get("qpu_probs") is not None:
        np.save(out_dir / "vqc_qpu_probs.npy", np.asarray(result["qpu_probs"], dtype=float))
    if result.get("sim_probs") is not None:
        np.save(out_dir / "vqc_qpu_sim_probs.npy", np.asarray(result["sim_probs"], dtype=float))
    if result.get("fakekingston_probs") is not None:
        np.save(
            out_dir / "vqc_fakekingston_probs.npy",
            np.asarray(result["fakekingston_probs"], dtype=float),
        )

    # ZNE comparison table: ideal sim vs noisy sim vs raw QPU vs ZNE-mitigated
    # on the ZNE subset.
    zne_probs = result.get("zne_probs")
    if zne_probs is not None:
        sim_probs = result.get("sim_probs")
        fk_probs = result.get("fakekingston_probs")
        qpu_probs = result.get("qpu_probs")
        csv_path = out_dir / "zne_comparison.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["sample_idx", "sim_prob", "fakekingston_prob", "raw_qpu_prob", "zne_prob"]
            )
            for i, zne_p in enumerate(np.asarray(zne_probs, dtype=float)):
                sim_p = float(sim_probs[i]) if sim_probs is not None and i < len(sim_probs) else ""
                fk_p = float(fk_probs[i]) if fk_probs is not None and i < len(fk_probs) else ""
                raw_p = float(qpu_probs[i]) if qpu_probs is not None and i < len(qpu_probs) else ""
                writer.writerow([i, sim_p, fk_p, raw_p, float(zne_p)])


def qpu_evaluate(cfg, circuit_qnode, params, X_qpu, y_qpu, backend=None) -> dict:
    """Orchestrate a QPU evaluation run.

    Mode is taken from ``cfg.run_mode``:

    * ``"sim"``  — ideal-simulator probabilities only (no hardware).
    * ``"qpu"``  — real-hardware inference + ZNE on a small sample.
    * ``"both"`` — hardware inference + ZNE **and** ideal-sim probabilities.

    The FakeKingston noisy-simulator baseline is always attempted (it is a
    local simulator and needs no token). Never crashes on missing hardware:
    every fatal step emits a warning, fills the corresponding key with
    ``None``, and is reported in the ``warnings`` list.

    Args:
        cfg: The pipeline ``Config`` instance.
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        X_qpu: Array of feature vectors, shape ``(n_samples, 64)``.
        y_qpu: Ground-truth labels (or ``None`` to skip metrics).
        backend: Optional pre-connected backend; when ``None`` it is resolved
            via ``get_ibm_backend(cfg)``.

    Returns:
        Dict with keys ``qpu_probs``, ``sim_probs``, ``zne_probs``,
        ``fakekingston_probs``, ``metrics``, ``backend_name``, ``warnings``.
    """
    warnings_list: list[str] = []
    result = {
        "qpu_probs": None,
        "sim_probs": None,
        "zne_probs": None,
        "fakekingston_probs": None,
        "metrics": None,
        "backend_name": None,
        "warnings": warnings_list,
    }
    mode = cfg.run_mode

    if X_qpu is None or len(X_qpu) == 0:
        warnings_list.append("X_qpu is empty; nothing to evaluate.")
        return result

    X = np.asarray(X_qpu, dtype=float)

    # --- Ideal-simulator probabilities ("sim" and "both" modes) -----------
    if mode in ("sim", "both"):
        try:
            from pipeline.vqc import vqc_predict

            result["sim_probs"] = np.asarray(
                vqc_predict(X, params, circuit_qnode), dtype=float
            )
        except Exception as exc:
            warnings_list.append(f"Simulator inference failed: {exc}")

    # --- Real-hardware inference ("qpu" and "both" modes) -----------------
    if mode in ("qpu", "both"):
        if backend is None:
            backend = get_ibm_backend(cfg)
        if backend is None:
            warnings_list.append("No IBM backend available; skipping QPU inference.")
        else:
            result["backend_name"] = getattr(backend, "name", str(backend))
            try:
                result["qpu_probs"] = run_vqc_on_qpu(
                    circuit_qnode, params, X, backend, shots=cfg.n_qpu_shots
                )
            except Exception as exc:
                warnings_list.append(f"QPU inference failed: {exc}")

            # ZNE on a small sample (first min(10, n) samples).
            try:
                executor = make_ibm_executor(backend, shots=cfg.n_qpu_shots)
                n_zne = min(10, len(X))
                zne_probs = np.empty(n_zne, dtype=float)
                for i in range(n_zne):
                    zne_probs[i] = run_zne_mitigation(
                        circuit_qnode,
                        params,
                        X[i],
                        backend,
                        executor,
                        scale_factors=cfg.zne_scale_factors,
                    )
                result["zne_probs"] = zne_probs
            except Exception as exc:
                warnings_list.append(f"ZNE mitigation failed: {exc}")

    # --- FakeKingston noisy-simulator baseline (always attempted) ---------
    try:
        result["fakekingston_probs"] = run_fakekingston_baseline(
            circuit_qnode, params, X, shots=cfg.n_qpu_shots
        )
    except Exception as exc:
        warnings_list.append(f"FakeKingston baseline failed: {exc}")

    # --- Metrics on the primary available probabilities -------------------
    primary = result["qpu_probs"] if result["qpu_probs"] is not None else result["sim_probs"]
    if primary is not None and y_qpu is not None:
        try:
            from pipeline.evaluate import compute_all_metrics

            result["metrics"] = compute_all_metrics(primary, np.asarray(y_qpu), tau=0.5)
        except Exception as exc:
            warnings_list.append(f"Metrics computation failed: {exc}")

    # --- Persist artifacts per plan Data Flow -----------------------------
    try:
        _save_artifacts(cfg, result)
    except Exception as exc:
        warnings_list.append(f"Failed to save QPU artifacts: {exc}")

    return result


__all__ = [
    "SEED",
    "resolve_ibm_token",
    "get_ibm_backend",
    "extract_z0_from_counts",
    "make_ibm_executor",
    "run_vqc_on_qpu",
    "run_zne_mitigation",
    "run_fakekingston_baseline",
    "qpu_evaluate",
]