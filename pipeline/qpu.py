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
    results/qpu_sample_indices.npy     original test-row indices of the balanced subset
    results/zne_comparison.csv         sim vs noisy-sim vs raw-QPU vs ZNE table
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import numpy as np

SEED = 6  # project-wide random seed (AGENTS.md global rule)

# SamplerV2 service cap for a single job: ~10M circuit executions (circuits x
# shots). Batching many pubs per job uses the device far more efficiently than
# one job per sample; the chunk size is derived from this cap instead of an
# arbitrary 64-circuit ceiling (which serialized the full test set across many
# jobs). A safety margin keeps jobs under the hard limit service-side.
_MAX_EXECUTIONS_PER_JOB = 10_000_000
_MAX_EXECUTIONS_SAFETY = 0.8
_QPU_TWIRL_RANDOMIZATIONS = 32  # Pauli twirling rounds (no usage increase: auto shot allocation)


def _job_chunk_size(n_circuits: int, shots: int, max_circuits_per_job: int = 0) -> int:
    """Per-SamplerV2-job circuit count when batching QPU inference / ZNE.

    ``max_circuits_per_job`` is an explicit cap (0 = auto). Auto derives it
    from the ~10M executions/job service limit: ``10M // shots`` scaled by the
    safety margin, capped at ``n_circuits``. Both batch loops (inference and
    ZNE) use this so a full test set fits in one job whenever possible.

    Args:
        n_circuits: Total number of circuits to submit.
        shots: Shots per circuit.
        max_circuits_per_job: Explicit circuit cap per job; 0 = auto-derive.

    Returns:
        Circuits per job, in ``[1, n_circuits]``.
    """
    if max_circuits_per_job and max_circuits_per_job > 0:
        return max(1, min(int(max_circuits_per_job), n_circuits))
    auto = int(_MAX_EXECUTIONS_PER_JOB // max(1, int(shots)) * _MAX_EXECUTIONS_SAFETY)
    return max(1, min(auto, n_circuits))


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
            channel=cfg.ibm_channel or None,
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

    Tries ``pennylane_qiskit.to_qiskit`` first (the plugin's classic API),
    then ``pennylane.qml.to_qiskit`` (re-exported in newer PennyLane), then a
    manual fallback for pennylane-qiskit >= 0.45 (which removed the
    top-level helper): a ``qnode.construct`` + Qiskit-gate-set decomposition
    + ``pennylane_qiskit.converter.circuit_to_qiskit`` pipeline mirroring
    ``QiskitDevice.preprocess``.
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
        pass

    # pennylane-qiskit >= 0.45: classic `to_qiskit` is gone; rebuild it from
    # the low-level converter used by QiskitDevice itself.
    try:
        import pennylane as qml
        from pennylane.devices.preprocess import decompose as pl_decompose
        from pennylane_qiskit.converter import QISKIT_OPERATION_MAP, circuit_to_qiskit
    except ImportError:
        return None

    operations = set(QISKIT_OPERATION_MAP.keys()) | {"GlobalPhase"}

    def converter(qnode):
        def convert_qnode(params, features):
            params = np.asarray(params, dtype=float)
            features = np.asarray(features, dtype=float)
            tape = qnode.construct((params, features), {})
            batch, _fn = pl_decompose(
                tape,
                target_gates=operations,
                stopping_condition=lambda op: op.name in operations,
                skip_initial_state_prep=False,
            )
            expanded = batch[0]
            return circuit_to_qiskit(
                expanded,
                register_size=len(expanded.wires),
                diagonalize=True,
                measure=True,
            )

        return convert_qnode

    return converter


def _strip_global_phase(qiskit_circuit):
    """Remove the ``global_phase`` op emitted by the pennylane-qiskit converter.

    ``pl_decompose`` leaves a trailing ``GlobalPhase`` gate on the tape (a
    remnant of the ``AmplitudeEmbedding`` -> ``StatePrep`` rewrite), which
    ``qiskit.qasm2.dumps`` refuses to export (``QASM2ExportError:
    OpenQASM 2 cannot represent 'global_phase'``). Mitiq's ZNE converts the
    circuit to Cirq through QASM2, so the op must be dropped. A global phase
    has no effect on the Pauli-Z expectation value that only the bitstring
    counts reduce to, so removal is measurement-safe.

    Args:
        qiskit_circuit: A Qiskit ``QuantumCircuit``.

    Returns:
        The same circuit with ``global_phase`` ops removed (mutated in place).
    """
    qiskit_circuit.global_phase = 0
    qiskit_circuit.data = [
        instr for instr in qiskit_circuit.data if instr.operation.name != "global_phase"
    ]
    return qiskit_circuit


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
    return _strip_global_phase(to_qiskit(circuit_qnode)(params, features))


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


def _make_sampler(backend, cfg=None):
    """Build a job-mode ``SamplerV2`` with optional DD + Pauli twirling.

    Dynamical decoupling (``XpXm`` sequence, middle slack distribution)
    suppresses idle-window dephasing; Pauli gate twirling (32 randomizations)
    converts coherent gate errors into stochastic noise without changing the
    expectation value. Both are SamplerV2 runtime options (IBM API contract
    names). Disabled when the corresponding ``cfg`` toggle is off or the
    runtime rejects the options (older qiskit-ibm-runtime) — never fatal.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    if cfg is None:
        return sampler
    try:
        if getattr(cfg, "qpu_dd_enable", False):
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.dynamical_decoupling.extra_slack_distribution = "middle"
        if getattr(cfg, "qpu_twirl_enable", False):
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = _QPU_TWIRL_RANDOMIZATIONS
            sampler.options.twirling.shots_per_randomization = "auto"
    except Exception as exc:
        warnings.warn(
            f"SamplerV2 runtime options rejected ({exc}); running without DD/twirling.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SamplerV2(mode=backend)
    return sampler


def make_ibm_executor(backend, shots=1024, cfg=None):
    """Build a mitiq-compatible executor: Qiskit circuit -> float ⟨Z₀⟩.

    The returned callable transpiles the circuit for *backend*
    (``optimization_level=3``), submits it via ``SamplerV2`` in job mode (no
    Session), and reduces the shot counts to a single ``⟨Z₀⟩`` value.

    Args:
        backend: A Qiskit ``BackendV2`` (e.g. from ``get_ibm_backend``).
        shots: Number of shots per circuit.
        cfg: Optional ``Config``; enables the DD + Pauli-twirling runtime
            options controlled by ``cfg.qpu_dd_enable`` /
            ``cfg.qpu_twirl_enable`` when provided.

    Returns:
        ``executor(circuit) -> float``.
    """

    def executor(circuit):
        from qiskit import transpile

        tqc = transpile(circuit, backend=backend, optimization_level=3)
        sampler = _make_sampler(backend, cfg)
        job = sampler.run([tqc], shots=shots)
        counts = _extract_counts(job.result()[0])
        return extract_z0_from_counts(counts, tqc.num_qubits)

    return executor


def run_vqc_on_qpu(circuit_qnode, params, X_subset, backend, shots=1024, cfg=None) -> np.ndarray | None:
    """Run the trained VQC on real IBM hardware for a subset of samples.

    All sample circuits are transpiled in one pass and submitted to
    ``SamplerV2`` in job mode, chunked at ``cfg.qpu_max_circuits_per_job``
    (0 = auto-derive from the ~10M executions/job service cap) instead of one
    job per sample. Results are read back in submission order.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        X_subset: Array of feature vectors, shape ``(n_samples, 64)``.
        backend: A Qiskit ``BackendV2`` (e.g. from ``get_ibm_backend``).
        shots: Number of shots per circuit (default 1024).
        cfg: Optional ``Config``; supplies the per-job circuit cap and the
            DD / Pauli-twirling runtime options (see ``_make_sampler``).

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
    circuits = [to_qiskit(circuit_qnode)(params, x) for x in X]
    tqcs = transpile(circuits, backend=backend, optimization_level=3)

    probs = np.empty(len(X), dtype=float)
    max_per_job = getattr(cfg, "qpu_max_circuits_per_job", 0) if cfg is not None else 0
    chunk_size = _job_chunk_size(len(tqcs), shots, max_per_job)
    sampler = _make_sampler(backend, cfg)
    for start in range(0, len(tqcs), chunk_size):
        chunk = tqcs[start : start + chunk_size]
        pub_results = sampler.run(chunk, shots=shots).result()
        for j, pub in enumerate(pub_results):
            counts = _extract_counts(pub)
            z0 = extract_z0_from_counts(counts, chunk[j].num_qubits)
            probs[start + j] = (1.0 + z0) / 2.0
    return probs


def run_zne_mitigation(
    circuit_qnode, params, x_sample, backend, executor, scale_factors=(1, 2, 3), cfg=None
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
        cfg: Optional ``Config`` forwarded to ``make_ibm_executor`` when a
            new executor must be built.

    Returns:
        The ZNE-mitigated ``⟨Z₀⟩``. If mitiq is missing, returns the raw
        noisy value from *executor* with a warning.
    """
    if executor is None:
        executor = make_ibm_executor(backend, cfg=cfg)

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


def run_zne_batched(
    circuit_qnode, params, X_subset, backend, scale_factors=(1, 2, 3), shots=1024, cfg=None
) -> np.ndarray | None:
    """Apply Zero-Noise Extrapolation to many samples in one SamplerV2 job.

    Folds every sample circuit at every noise scale factor with mitiq's
    ``fold_gates_at_random``, transpiles the whole batch in one pass and
    submits it as a single job (chunked at ``cfg.qpu_max_circuits_per_job``,
    0 = auto from the ~10M executions/job cap), then Richardson-extrapolates
    per sample. This replaces the per-sample ``execute_with_zne`` flow (3 jobs
    per sample) with one job total.

    Args:
        circuit_qnode: The ``qml.QNode`` built by
            ``pipeline.ansatz.build_vqc_circuit``.
        params: Flat trainable parameter array (``split_params`` layout).
        X_subset: Array of feature vectors, shape ``(n_samples, 64)``.
        backend: A Qiskit ``BackendV2`` (e.g. from ``get_ibm_backend``).
        scale_factors: Noise scale factors for Richardson extrapolation.
        shots: Number of shots per circuit (default 1024).
        cfg: Optional ``Config``; supplies the per-job circuit cap and the
            DD / Pauli-twirling runtime options (see ``_make_sampler``).

    Returns:
        Array of ZNE-mitigated probabilities ``(1 + ⟨Z₀⟩) / 2``, shape
        ``(n_samples,)``, or ``None`` if mitiq / qiskit are unavailable.
    """
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2
    except ImportError:
        warnings.warn(
            "qiskit / qiskit-ibm-runtime not installed; cannot run ZNE on QPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        from mitiq.zne.inference import RichardsonFactory
        from mitiq.zne.scaling import fold_gates_at_random
    except ImportError:
        warnings.warn(
            "mitiq not installed; cannot run batched ZNE.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    params = np.asarray(params, dtype=float)
    X = np.asarray(X_subset, dtype=float)
    sfs = list(scale_factors)

    jobs = []
    for i, x in enumerate(X):
        qc = _qnode_to_qiskit(circuit_qnode, params, x)
        for k, sf in enumerate(sfs):
            jobs.append((i, k, fold_gates_at_random(qc, sf)))

    tqcs = transpile([jc for _, _, jc in jobs], backend=backend, optimization_level=3)

    z0 = np.empty((len(X), len(sfs)), dtype=float)
    max_per_job = getattr(cfg, "qpu_max_circuits_per_job", 0) if cfg is not None else 0
    chunk_size = _job_chunk_size(len(tqcs), shots, max_per_job)
    sampler = _make_sampler(backend, cfg)
    for start in range(0, len(tqcs), chunk_size):
        chunk = tqcs[start : start + chunk_size]
        pub_results = sampler.run(chunk, shots=shots).result()
        for j, pub in enumerate(pub_results):
            sample_idx, scale_idx, _ = jobs[start + j]
            counts = _extract_counts(pub)
            z0[sample_idx, scale_idx] = extract_z0_from_counts(counts, chunk[j].num_qubits)

    probs = np.empty(len(X), dtype=float)
    for i in range(len(X)):
        mitigated_z0 = RichardsonFactory.extrapolate(sfs, z0[i].tolist())
        probs[i] = (1.0 + mitigated_z0) / 2.0
    return probs


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

    # Heron r2 fakes live in different modules across qiskit-ibm-runtime
    # versions; try the modern location first, then the legacy one.
    # FakeKingston was removed in qiskit-ibm-runtime >= 0.45; FakeFez and
    # FakeMarrakesh are the equivalent Heron r2 devices there.
    fake_cls = None
    for _name in ("FakeKingston", "FakeFez", "FakeMarrakesh"):
        try:
            from qiskit_ibm_runtime.fake_provider import fake_provider as _fp

            fake_cls = getattr(_fp, _name, None)
            if fake_cls is not None:
                break
        except ImportError:
            continue
    if fake_cls is None:
        try:
            from qiskit.providers.fake_provider import FakeKingston

            fake_cls = FakeKingston
        except ImportError:
            fake_cls = None
    if fake_cls is None:
        warnings.warn(
            "No Heron r2 fake backend available; skipping noisy baseline.",
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

    noise_model = NoiseModel.from_backend(fake_cls())
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


def select_qpu_subset(X, y, n_samples, seed=SEED):
    """Select a class-balanced, reproducible QPU evaluation subset.

    The project's test split is label-sorted (all negatives precede all
    positives), so a naive ``X[:n]`` slice evaluates a single class and
    makes AUC / sensitivity undefined — the bug that corrupted the
    2026-09-15 ``ibm_miami`` run (79 jobs on an all-negative slice). This
    helper performs stratified sampling — up to ``ceil(n/2)`` positives and
    ``floor(n/2)`` negatives, topped up from the minority-direction side
    when one class is under-populated — using the project-wide seed, and
    returns the **row indices** so the job → sample mapping can always be
    reconstructed from ``results/qpu_sample_indices.npy``.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: Label vector of shape ``(n_samples,)``.
        n_samples: Target subset size (capped at ``len(y)``).
        seed: Random seed (project-wide ``SEED = 6`` by default).

    Returns:
        Tuple ``(X_sub, y_sub, idx)`` where the selected rows are returned
        in *submission order* (seeded-shuffled) and ``idx`` are the original
        row indices (same order) in ``X`` / ``y``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = min(int(n_samples), len(y))
    if n <= 0:
        raise ValueError("n_samples must be a positive integer")

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    n_pos_target = int(np.ceil(n / 2))
    n_neg_target = n - n_pos_target
    n_pos = min(n_pos_target, len(pos))
    n_neg = min(n_neg_target, len(neg))
    if n_pos + n_neg < n:  # one class under-populated → top up from the other
        shortfall = n - (n_pos + n_neg)
        if n_neg < n_neg_target and len(pos) > n_pos:
            take = min(shortfall, len(pos) - n_pos)
            n_pos += take
            shortfall -= take
        if shortfall > 0 and len(neg) > n_neg:
            n_neg += min(shortfall, len(neg) - n_neg)

    pos_idx = rng.choice(pos, size=n_pos, replace=False)
    neg_idx = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)  # submission order — not class-blocked
    return X[idx], y[idx], idx


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
                    circuit_qnode, params, X, backend, shots=cfg.n_qpu_shots, cfg=cfg
                )
            except Exception as exc:
                warnings_list.append(f"QPU inference failed: {exc}")

            # ZNE on a small sample (first min(10, n) samples), batched into
            # a single job instead of one job per sample.
            try:
                n_zne = min(10, len(X))
                result["zne_probs"] = run_zne_batched(
                    circuit_qnode,
                    params,
                    X[:n_zne],
                    backend,
                    scale_factors=cfg.zne_scale_factors,
                    shots=cfg.n_qpu_shots,
                    cfg=cfg,
                )
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

            result["metrics"] = compute_all_metrics(
                primary, np.asarray(y_qpu), tau=cfg.qpu_tau
            )
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
    "run_zne_batched",
    "run_fakekingston_baseline",
    "qpu_evaluate",
]