from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

try:
    from nnse_accessibility import choose_biggest_neutral_npz
except ModuleNotFoundError:  # Allows package-style imports in local tests.
    from .nnse_accessibility import choose_biggest_neutral_npz


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "sloppy_subspace"
RESULTS.mkdir(parents=True, exist_ok=True)


TYSON_CANONICAL = {
    "k1_aa_over_CT": 0.015,
    "k2": 0.0,
    "k3_CT": 200.0,
    "k4": 180.0,
    "k4prime": 0.018,
    "k5_minusP": 0.0,
    "k6": 1.0,
    "k7": 0.6,
    "k8_minusP": 100.0,
    "k9": 50.0,
    "CT": 1.0,
}

TYSON_NAME_MAP = {
    "k1aa": "k1_aa_over_CT",
    "k1_aa_over_CT": "k1_aa_over_CT",
    "k3": "k3_CT",
    "k3_CT": "k3_CT",
    "k4": "k4",
    "k4prime": "k4prime",
    "k6": "k6",
    "k7": "k7",
    "k8notP": "k8_minusP",
    "k8_minusP": "k8_minusP",
    "k9": "k9",
}

TYSON_Y0 = np.array([0.9, 0.05, 0.0, 0.005, 0.3, 0.0], dtype=float)


@dataclass
class SloppySubspaceConfig:
    model: str = "tyson1991"
    npz: str | None = None
    neutral_threshold: float = 15.0
    n_points: int = 80
    k_sloppy: int = 3
    local_neighbors: int = 12
    seed: int = 42
    t_end: float = 100.0
    n_time: int = 501
    tag: str | None = None


def f_m(m: float, p: dict[str, float]) -> float:
    return p["k4prime"] + p["k4"] * (m / p["CT"]) ** 2


def df_m_dm(m: float, p: dict[str, float]) -> float:
    return p["k4"] * 2.0 * (m / p["CT"]) * (1.0 / p["CT"])


def tyson_rhs(_t: float, x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    c2, cp, pm, m, y, yp = x
    k3 = p["k3_CT"] / p["CT"]
    k1 = p["k1_aa_over_CT"] * p["CT"]
    dc2 = p["k6"] * m - p["k8_minusP"] * c2 + p["k9"] * cp
    dcp = -k3 * cp * y + p["k8_minusP"] * c2 - p["k9"] * cp
    dpm = k3 * cp * y - pm * f_m(m, p) + p["k5_minusP"] * m
    dm = pm * f_m(m, p) - p["k5_minusP"] * m - p["k6"] * m
    dy = k1 - p["k2"] * y - k3 * cp * y
    dyp = p["k6"] * m - p["k7"] * yp
    return np.array([dc2, dcp, dpm, dm, dy, dyp], dtype=float)


def jacobian_fx(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    c2, cp, pm, m, y, _yp = x
    k3 = p["k3_CT"] / p["CT"]
    df = df_m_dm(m, p)
    j = np.zeros((6, 6), dtype=float)
    j[0, 0] = -p["k8_minusP"]
    j[0, 1] = p["k9"]
    j[0, 3] = p["k6"]
    j[1, 0] = p["k8_minusP"]
    j[1, 1] = -k3 * y - p["k9"]
    j[1, 4] = -k3 * cp
    j[2, 1] = k3 * y
    j[2, 2] = -f_m(m, p)
    j[2, 3] = -pm * df + p["k5_minusP"]
    j[2, 4] = k3 * cp
    j[3, 2] = f_m(m, p)
    j[3, 3] = pm * df - p["k5_minusP"] - p["k6"]
    j[4, 1] = -k3 * y
    j[4, 4] = -p["k2"] - k3 * cp
    j[5, 3] = p["k6"]
    j[5, 5] = -p["k7"]
    return j


def df_dparam(x: np.ndarray, p: dict[str, float], param_names: list[str]) -> np.ndarray:
    c2, cp, pm, m, y, yp = x
    dfdh = np.zeros((6, len(param_names)), dtype=float)
    for col, name in enumerate(param_names):
        df = np.zeros(6, dtype=float)
        if name == "k1_aa_over_CT":
            df[4] = p["CT"]
        elif name == "k3_CT":
            scale = 1.0 / p["CT"]
            df[1] = -scale * cp * y
            df[2] = scale * cp * y
            df[4] = -scale * cp * y
        elif name == "k4":
            scale = (m / p["CT"]) ** 2
            df[2] = -pm * scale
            df[3] = pm * scale
        elif name == "k4prime":
            df[2] = -pm
            df[3] = pm
        elif name == "k6":
            df[0] = m
            df[3] = -m
            df[5] = m
        elif name == "k7":
            df[5] = -yp
        elif name == "k8_minusP":
            df[0] = -c2
            df[1] = c2
        elif name == "k9":
            df[0] = cp
            df[1] = -cp
        else:
            raise KeyError(f"Unsupported Tyson Hessian parameter: {name}")
        dfdh[:, col] = df
    return dfdh


def map_tyson_parameter_names(names: list[str]) -> tuple[list[int], list[str], list[str]]:
    supported_idx: list[int] = []
    mapped: list[str] = []
    unsupported: list[str] = []
    for idx, name in enumerate(names):
        if name in TYSON_NAME_MAP:
            supported_idx.append(idx)
            mapped.append(TYSON_NAME_MAP[name])
        else:
            unsupported.append(name)
    return supported_idx, mapped, unsupported


def vector_to_tyson_params(vector: np.ndarray, source_names: list[str]) -> tuple[dict[str, float], list[int], list[str], list[str]]:
    supported_idx, mapped_names, unsupported = map_tyson_parameter_names(source_names)
    p = dict(TYSON_CANONICAL)
    for idx, mapped in zip(supported_idx, mapped_names):
        p[mapped] = float(vector[idx])
    return p, supported_idx, mapped_names, unsupported


def compute_tyson_hessian(
    vector: np.ndarray,
    source_names: list[str],
    t_end: float = 100.0,
    n_time: int = 501,
) -> dict:
    p, supported_idx, mapped_names, unsupported = vector_to_tyson_params(vector, source_names)
    if len(mapped_names) < 2:
        raise ValueError("Need at least two supported Tyson parameters for subspace analysis")
    p_dim = len(mapped_names)
    t_eval = np.linspace(0.0, t_end, n_time)

    def aug_rhs(t: float, z: np.ndarray) -> np.ndarray:
        x = z[:6]
        sensitivities = z[6:].reshape((6, p_dim), order="F")
        xdot = tyson_rhs(t, x, p)
        a = jacobian_fx(x, p)
        dfdh = df_dparam(x, p, mapped_names)
        dfdlogh = np.zeros_like(dfdh)
        for j, name in enumerate(mapped_names):
            dfdlogh[:, j] = p[name] * dfdh[:, j]
        sdot = a.dot(sensitivities) + dfdlogh
        return np.concatenate([xdot, sdot.ravel(order="F")])

    z0 = np.concatenate([TYSON_Y0, np.zeros((6, p_dim)).ravel(order="F")])
    sol = solve_ivp(aug_rhs, (0.0, t_end), z0, method="BDF", t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError("Augmented Tyson integration failed: " + sol.message)

    tt = sol.t
    dt = np.diff(tt)
    q = np.zeros_like(tt)
    q[0] = dt[0] / 2.0
    q[-1] = dt[-1] / 2.0
    q[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    weight = np.diag(np.ones(6) / (6.0 * t_end))

    hessian = np.zeros((p_dim, p_dim), dtype=float)
    s_flat = sol.y[6:, :]
    for i in range(sol.t.size):
        jac = s_flat[:, i].reshape((6, p_dim), order="F")
        hessian += q[i] * jac.T.dot(weight).dot(jac)

    eigvals, eigvecs = np.linalg.eigh(hessian)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    return {
        "H": hessian,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "supported_indices": supported_idx,
        "mapped_names": mapped_names,
        "unsupported_names": unsupported,
    }


def sloppy_basis(eigvecs_descending: np.ndarray, k: int) -> np.ndarray:
    k = min(k, eigvecs_descending.shape[1])
    basis = eigvecs_descending[:, -k:]
    q, _ = np.linalg.qr(basis)
    return q[:, :k]


def principal_angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(u.T @ v, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


def chordal_distance(u: np.ndarray, v: np.ndarray) -> float:
    singular = np.linalg.svd(u.T @ v, compute_uv=False)
    return float(np.sqrt(max(0.0, min(u.shape[1], v.shape[1]) - np.sum(singular * singular))))


def load_neutral_points(npz_path: Path, neutral_threshold: float = 15.0) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "neutral_points" in data.files:
        points = np.asarray(data["neutral_points"], dtype=float)
        values = np.asarray(data["neutral_objective_values"], dtype=float)
        source = "neutral_points"
    elif "candidate_vectors" in data.files:
        all_points = np.asarray(data["candidate_vectors"], dtype=float)
        all_values = np.asarray(data["candidate_objective_values"], dtype=float)
        keep = np.isfinite(all_values) & (all_values <= neutral_threshold)
        points = all_points[keep]
        values = all_values[keep]
        source = f"candidate_vectors<= {neutral_threshold:g}"
    else:
        raise KeyError(f"{npz_path} contains neither neutral_points nor candidate_vectors")
    if len(values) != len(points):
        n = min(len(values), len(points))
        points = points[:n]
        values = values[:n]
    return {
        "points": points,
        "values": values,
        "parameter_names": [str(x) for x in data["parameter_names"]],
        "p0": np.asarray(data["p0"], dtype=float),
        "bin_thresholds": np.asarray(data["bin_thresholds"], dtype=float),
        "source": source,
    }


def select_points(points: np.ndarray, n_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    finite = np.all(np.isfinite(points), axis=1) & np.all(points > 0, axis=1)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        raise ValueError("No finite positive neutral points available")
    if len(idx) > n_points:
        idx = rng.choice(idx, size=n_points, replace=False)
    return idx, points[idx]


def local_tangent_bases(log_coords: np.ndarray, k: int, neighbors: int) -> np.ndarray:
    n, dim = log_coords.shape
    k = min(k, dim)
    neighbors = min(max(k + 1, neighbors), n)
    tree = cKDTree(log_coords)
    bases = np.full((n, dim, k), np.nan, dtype=float)
    for i in range(n):
        _, idx = tree.query(log_coords[i], k=neighbors)
        idx = np.atleast_1d(idx)
        local = log_coords[idx] - log_coords[i]
        _, _, vt = np.linalg.svd(local, full_matrices=False)
        basis = vt[:k].T
        q, _ = np.linalg.qr(basis)
        bases[i] = q[:, :k]
    return bases


def analyze_tyson_sloppy_subspaces(config: SloppySubspaceConfig) -> dict:
    if config.model != "tyson1991":
        raise NotImplementedError("The current Hessian/subspace implementation is Tyson-specific")
    npz_path = Path(config.npz) if config.npz else choose_biggest_neutral_npz(ROOT, config.model)
    loaded = load_neutral_points(npz_path, config.neutral_threshold)
    rng = np.random.default_rng(config.seed)
    selected_idx, selected = select_points(loaded["points"], config.n_points, rng)

    first = compute_tyson_hessian(selected[0], loaded["parameter_names"], config.t_end, config.n_time)
    supported_idx = first["supported_indices"]
    mapped_names = first["mapped_names"]
    unsupported_names = first["unsupported_names"]
    supported = selected[:, supported_idx]
    log_coords = np.log(supported)
    p0_supported = loaded["p0"][supported_idx]
    log_wt = np.log(p0_supported)

    eigvals = []
    bases = []
    failed = []
    start = time.time()
    for i, vector in enumerate(selected):
        try:
            res = compute_tyson_hessian(vector, loaded["parameter_names"], config.t_end, config.n_time)
            eigvals.append(res["eigvals"])
            bases.append(sloppy_basis(res["eigvecs"], config.k_sloppy))
        except Exception as exc:
            failed.append({"selection_index": int(i), "neutral_index": int(selected_idx[i]), "error": str(exc)})
            eigvals.append(np.full(len(mapped_names), np.nan))
            bases.append(np.full((len(mapped_names), min(config.k_sloppy, len(mapped_names))), np.nan))

    valid = np.array([np.all(np.isfinite(basis)) for basis in bases], dtype=bool)
    bases_arr = np.asarray(bases, dtype=float)
    eigvals_arr = np.asarray(eigvals, dtype=float)

    wt_res = compute_tyson_hessian(loaded["p0"], loaded["parameter_names"], config.t_end, config.n_time)
    wt_basis = sloppy_basis(wt_res["eigvecs"], config.k_sloppy)

    valid_indices = np.where(valid)[0]
    n_valid = len(valid_indices)
    pairwise_chordal = np.full((n_valid, n_valid), np.nan, dtype=float)
    pairwise_max_angle = np.full((n_valid, n_valid), np.nan, dtype=float)
    for a_pos, a in enumerate(valid_indices):
        for b_pos, b in enumerate(valid_indices):
            if b_pos < a_pos:
                pairwise_chordal[a_pos, b_pos] = pairwise_chordal[b_pos, a_pos]
                pairwise_max_angle[a_pos, b_pos] = pairwise_max_angle[b_pos, a_pos]
                continue
            angles = principal_angles(bases_arr[a], bases_arr[b])
            pairwise_chordal[a_pos, b_pos] = chordal_distance(bases_arr[a], bases_arr[b])
            pairwise_max_angle[a_pos, b_pos] = float(np.max(angles))

    wt_chordal = np.full(len(selected), np.nan, dtype=float)
    wt_max_angle = np.full(len(selected), np.nan, dtype=float)
    for i in valid_indices:
        angles = principal_angles(bases_arr[i], wt_basis)
        wt_chordal[i] = chordal_distance(bases_arr[i], wt_basis)
        wt_max_angle[i] = float(np.max(angles))

    tangent_bases = local_tangent_bases(log_coords, config.k_sloppy, config.local_neighbors)
    tangent_chordal = np.full(len(selected), np.nan, dtype=float)
    tangent_max_angle = np.full(len(selected), np.nan, dtype=float)
    for i in valid_indices:
        angles = principal_angles(bases_arr[i], tangent_bases[i])
        tangent_chordal[i] = chordal_distance(bases_arr[i], tangent_bases[i])
        tangent_max_angle[i] = float(np.max(angles))

    log_distance_to_wt = np.linalg.norm(log_coords - log_wt[None, :], axis=1)
    elapsed = time.time() - start
    summary = {
        "config": asdict(config),
        "npz": str(npz_path),
        "point_source": loaded["source"],
        "neutral_points_in_file": int(len(loaded["points"])),
        "neutral_points_sampled": int(len(selected)),
        "valid_hessians": int(n_valid),
        "failed_hessians": failed,
        "supported_parameter_names": mapped_names,
        "unsupported_parameter_names": unsupported_names,
        "elapsed_seconds": elapsed,
        "pairwise_chordal_median": float(np.nanmedian(pairwise_chordal)) if n_valid else math.nan,
        "wt_chordal_median": float(np.nanmedian(wt_chordal)) if n_valid else math.nan,
        "tangent_chordal_median": float(np.nanmedian(tangent_chordal)) if n_valid else math.nan,
    }
    return {
        "summary": summary,
        "selected_indices": selected_idx,
        "selected_points": selected,
        "supported_points": supported,
        "supported_parameter_names": np.asarray(mapped_names, dtype=object),
        "unsupported_parameter_names": np.asarray(unsupported_names, dtype=object),
        "eigvals": eigvals_arr,
        "bases": bases_arr,
        "valid": valid,
        "pairwise_chordal": pairwise_chordal,
        "pairwise_max_angle": pairwise_max_angle,
        "wt_chordal": wt_chordal,
        "wt_max_angle": wt_max_angle,
        "tangent_chordal": tangent_chordal,
        "tangent_max_angle": tangent_max_angle,
        "log_distance_to_wt": log_distance_to_wt,
        "log_coords": log_coords,
    }


def save_sloppy_subspace(result: dict, tag: str) -> tuple[Path, Path]:
    npz_path = RESULTS / f"{tag}.npz"
    json_path = RESULTS / f"{tag}.json"
    arrays = {k: v for k, v in result.items() if k != "summary"}
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(result["summary"], indent=2))
    return npz_path, json_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare local sloppy Hessian subspaces across a Tyson NNSE neutral set")
    parser.add_argument("--model", default="tyson1991")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--neutral-threshold", type=float, default=15.0)
    parser.add_argument("--n-points", type=int, default=80)
    parser.add_argument("--k-sloppy", type=int, default=3)
    parser.add_argument("--local-neighbors", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-end", type=float, default=100.0)
    parser.add_argument("--n-time", type=int, default=501)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    config = SloppySubspaceConfig(**vars(args))
    result = analyze_tyson_sloppy_subspaces(config)
    tag = args.tag or f"{args.model}_sloppy_subspace_n{args.n_points}_k{args.k_sloppy}"
    out_npz, out_json = save_sloppy_subspace(result, tag)
    print(f"Saved {out_npz}")
    print(f"Saved {out_json}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
