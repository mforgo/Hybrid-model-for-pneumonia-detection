"""Render the VQC as it appears AFTER IBM transpilation, in IBM Quantum style.

Builds the LOCKED VQC structure (pipeline/ansatz.py, build_vqc_circuit):

    AmplitudeEmbedding(64-dim, normalize)          # once
    3 × [ RY(scale_q * x_q * pi, q)                # data re-upload
          Rot(phi, theta, omega, q)                # = U3
          CNOT ring q -> q+1 ]                     # entangler
    RY(meas0, 0); RZ(meas1, 0)                     # trainable measurement basis
    measure all -> <Z0> from bitstrings

natively in Qiskit (initialize = amplitude embedding, U3 = Rot), then transpiles
for a Heron r2 device (FakeMarrakesh, 156 qubits) at optimization_level=3, exactly
as `pipeline/qpu.py` does for real hardware runs.

Outputs (IBM Quantum 'iqx' draw style, vector SVG):
    media/vqc_transpiled_full.svg      - entire transpiled circuit, folded
    media/vqc_transpiled_fragment.svg  - first K operations, unfolded wide strip
                                         (legible close-up: native ECR/RZ/SX + routing)

Parameter VALUES are cosmetic for the diagram; the gate structure is what matters.
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

SEED = 6
N_QUBITS = 6
N_LAYERS = 3
OPT_LEVEL = 3
SHOTS_PLACEHOLDER = 1024  # not used for drawing; documented for parity

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")


def _feature_vector(n_amp: int = 2**N_QUBITS) -> np.ndarray:
    """Deterministic 64-dim L2-normalised feature vector (seed 6)."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(n_amp)
    return x / np.linalg.norm(x)


def _params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rot, scale, meas) from results/vqc_best_params.npy if present.

    The saved file holds the OLD 54-param layout (rot only, shape (3,6,3));
    scale/meas are filled with neutral values — irrelevant for the diagram.
    """
    rot_path = os.path.join(PROJECT_ROOT, "results", "vqc_best_params.npy")
    rot = np.zeros((N_LAYERS, N_QUBITS, 3))
    if os.path.exists(rot_path):
        arr = np.load(rot_path)
        if arr.shape == (N_LAYERS, N_QUBITS, 3):
            rot = arr
    scale = np.ones(N_QUBITS)
    meas = np.zeros(2)
    return rot, scale, meas


def build_vqc_qiskit() -> "QuantumCircuit":
    """The LOCKED VQC structure as a raw Qiskit circuit (pre-transpile)."""
    from qiskit import QuantumCircuit

    rot, scale, meas = _params()
    x = _feature_vector()

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)

    # AmplitudeEmbedding once (equivalent state preparation).
    qc.initialize(x, range(N_QUBITS))

    # Data re-uploading layers.
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.ry(scale[q] * x[q] * np.pi, q)
        for q in range(N_QUBITS):
            qc.u(rot[l, q, 0], rot[l, q, 1], rot[l, q, 2], q)  # Rot = U3
        for q in range(N_QUBITS):
            qc.cx(q, (q + 1) % N_QUBITS)

    # Trainable measurement basis on qubit 0.
    qc.ry(meas[0], 0)
    qc.rz(meas[1], 0)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def _backend():
    """Heron r2 fake backend (fallback chain matches pipeline/qpu.py)."""
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    return FakeMarrakesh()


def transpile_for_device(qc, backend, n_qubits: int = N_QUBITS):
    """Transpile for the device with a 6-qubit connected subgraph layout."""
    from qiskit import transpile

    # Pick 6 connected device qubits: BFS chain 0-1-2-3-4-5 if present in the
    # coupling map (transpiler keeps virtual->physical mapping via layouter).
    cm = backend.coupling_map
    chain = list(range(n_qubits))
    if all((i, i + 1) in cm.get_edges() for i in range(n_qubits - 1)):
        initial_layout = chain
    else:
        # Fallback: first connected 6-qubit path found by greedy BFS.
        adj: dict[int, list[int]] = {}
        for a, b in cm.get_edges():
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        seen, queue = {0}, [0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            if len(order) == n_qubits:
                break
            for nb in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        initial_layout = order

    return transpile(
        qc,
        backend=backend,
        optimization_level=OPT_LEVEL,
        initial_layout=initial_layout,
    )


def compactify(tqc):
    """Remap a device-wide transpiled circuit onto only its used wires.

    Qiskit keeps the full target-backend register (156 qubits for
    FakeMarrakesh); the drawer would render 150 empty wires. This rebuilds
    the circuit over the physical qubits/clbits that actually carry ops.
    """
    from qiskit import QuantumCircuit

    qubits_used = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})
    clbits_used = sorted({tqc.find_bit(c).index for inst in tqc.data for c in inst.clbits})
    out = QuantumCircuit(len(qubits_used), len(clbits_used))
    out.metadata = dict(tqc.metadata)
    qmap = {phys: i for i, phys in enumerate(qubits_used)}
    cmap = {phys: i for i, phys in enumerate(clbits_used)}
    for inst in tqc.data:
        qargs = [qmap[tqc.find_bit(q).index] for q in inst.qubits]
        cargs = [cmap[tqc.find_bit(c).index] for c in inst.clbits]
        out.append(inst.operation, qargs, cargs)
    return out


def slice_circuit(tqc, n_ops: int):
    """Return the first ``n_ops`` instructions of a transpiled circuit."""
    from qiskit import QuantumCircuit

    if n_ops >= len(tqc.data) or n_ops <= 0:
        n_ops = len(tqc.data)
    out = QuantumCircuit(tqc.num_qubits, tqc.num_clbits)
    out.metadata = dict(tqc.metadata)
    for inst in tqc.data[:n_ops]:
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def draw(circuit, path: str, fold=None, figsize=None):
    """Draw with IBM Quantum Platform ('iqp') style and save as SVG."""
    import matplotlib

    matplotlib.rcParams["svg.fonttype"] = "none"  # keep labels as editable <text>
    from qiskit.visualization import circuit_drawer

    style: dict = {"name": "iqp"}
    if figsize is not None:
        style["figsize"] = figsize  # MPL drawer accepts figsize only via style dict
    fig = circuit_drawer(circuit, output="mpl", style=style, fold=fold)
    fig.savefig(path, bbox_inches="tight", transparent=False)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  saved {path} ({os.path.getsize(path):,} bytes)")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MEDIA_DIR, exist_ok=True)

    print("== building locked VQC structure in Qiskit ==")
    qc = build_vqc_qiskit()
    ops = qc.count_ops()
    print(f"  pre-transpile ops: {dict(ops)}  (total {sum(ops.values())})")

    print("== transpiling for FakeMarrakesh (Heron r2, opt level 3) ==")
    backend = _backend()
    tqc = transpile_for_device(qc, backend)
    tqc = compactify(tqc)
    tops = tqc.count_ops()
    depth = tqc.depth()
    n2q = sum(c for gate, c in tops.items() if gate in ("ecr", "cx", "cz", "swap"))
    print(
        f"  post-transpile ops: {dict(tops)}  (total {sum(tops.values())}, "
        f"2-qubit {n2q}, depth {depth})"
    )
    print(f"  circuit width: {tqc.num_qubits} qubits / {tqc.num_clbits} clbits")

    print("== drawing (IBM 'iqx' style) ==")
    full_path = os.path.join(MEDIA_DIR, "vqc_transpiled_full.svg")
    frag_path = os.path.join(MEDIA_DIR, "vqc_transpiled_fragment.svg")

    # Full circuit, folded to a poster-friendly column width.
    draw(tqc, full_path, fold=15, figsize=(9, 10))

    # Legible close-up: first ~1.5 state-prep equivalents + layer 1, unfolded.
    frag_ops = min(90, len(tqc.data))
    frag = slice_circuit(tqc, frag_ops)
    draw(frag, frag_path, fold=None, figsize=(24, 6))

    print("\n== done ==")


if __name__ == "__main__":
    sys.exit(main())