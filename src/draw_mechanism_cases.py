"""Draw mechanism case visualizations for EWGB-TAD.

The figure uses controlled synthetic trajectories so that the selected anomaly
types are known. It visualizes the selected reference region, case-specific
feature contributions to its weighted distance, and view-wise score
contributions for representative cases.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_generator import generate_synthetic_trajectories
from ewgb_tad_current import ThreeViewEWGBDetector
from feature_extraction_v2 import (
    KINEMATIC_NAMES,
    PATH_DEVIATION_NAMES,
    SPATIAL_NAMES,
    extract_all_features_v2,
)


PAPER_DIR = Path(os.environ.get("EWGB_AUX_PAPER_DIR", str(ROOT / "paper_work")))

VIEW_LABELS = {
    "spatial_path": "Spatial-path",
    "kinematic": "Kinematic",
    "trajectory_shape": "Shape",
}
VIEW_COLORS = {
    "spatial_path": "#4C78A8",
    "kinematic": "#F28E2B",
    "trajectory_shape": "#B07AA1",
}
CASE_HEADERS = {
    "Normal trajectory": "Normal case",
    "Speed anomaly": "Speed anomaly",
    "Detour anomaly": "Detour anomaly",
    "Loop anomaly": "Loop anomaly",
    "Route-deviation anomaly": "Route deviation",
}
CASE_COLORS = {
    "Normal trajectory": "#4C78A8",
    "Speed anomaly": "#F28E2B",
    "Detour anomaly": "#E15759",
    "Loop anomaly": "#E15759",
    "Route-deviation anomaly": "#E15759",
}
CASE_SPEED_CMAPS = {
    "Normal trajectory": LinearSegmentedColormap.from_list("normal_speed", ["#1F4E79", "#D8E6F3"]),
    "Speed anomaly": LinearSegmentedColormap.from_list(
        "speed_anomaly_speed",
        ["#7F2704", "#D95F0E", "#FEE6CE"],
    ),
    "Detour anomaly": LinearSegmentedColormap.from_list("structural_speed", ["#B2182B", "#F4B6B2"]),
    "Loop anomaly": LinearSegmentedColormap.from_list("loop_speed", ["#B2182B", "#F4B6B2"]),
    "Route-deviation anomaly": LinearSegmentedColormap.from_list("route_speed", ["#B2182B", "#F4B6B2"]),
}
FEATURE_NAMES = {
    "spatial_path": SPATIAL_NAMES + PATH_DEVIATION_NAMES,
    "kinematic": KINEMATIC_NAMES,
    "trajectory_shape": [f"shape_PC{i}" for i in range(1, 11)],
}
SHORT_FEATURE_NAMES = {
    "total_length": "length",
    "od_distance": "OD dist.",
    "detour_ratio": "detour",
    "mean_lateral_dev": "mean lat.",
    "max_lateral_dev": "max lat.",
    "bbox_area": "bbox",
    "radius_of_gyration": "gyration",
    "mean_dir_consistency": "mean dir.",
    "min_dir_consistency": "min dir.",
    "second_half_consistency": "late dir.",
    "cum_angle_dev": "cum angle",
    "max_angle_dev": "max angle",
    "curvature_var": "curv. var",
    "curvature_max": "max curv.",
    "backward_ratio": "backward",
    "max_local_detour": "local detour",
    "mean_speed": "mean speed",
    "std_speed": "std speed",
    "max_speed": "max speed",
    "mean_accel": "mean accel.",
    "std_accel": "std accel.",
    "stop_ratio": "stop ratio",
    "mean_turn": "mean turn",
    "max_turn": "max turn",
}

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.65,
    }
)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    smin, smax = float(scores.min()), float(scores.max())
    if smax - smin <= 1e-8:
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)


def _view_data(detector: ThreeViewEWGBDetector, spatial_path, kinematic, trajs):
    flat = np.array([traj.flatten() for traj in trajs])
    if detector.pca is None:
        raise RuntimeError("Detector must be fitted before drawing mechanism cases.")
    shape = detector.pca.transform(flat)
    return {
        "spatial_path": spatial_path,
        "kinematic": kinematic,
        "trajectory_shape": shape,
    }


def _nearest_ball(det, row: np.ndarray) -> tuple[int, float]:
    norm = (row - det.mean) / det.std
    centers = np.array([ball.center for ball in det.balls])
    weights = np.array(det.local_weights)
    radii = np.array([ball.radius for ball in det.balls])
    densities = np.array([ball.density() for ball in det.balls])
    diffs = norm[np.newaxis, :] - centers
    weighted_dist = np.sqrt(np.sum(weights * diffs**2, axis=1))
    weighted_dist = np.where(weighted_dist < radii, weighted_dist * 0.5, weighted_dist)
    ball_scores = weighted_dist / (np.log1p(densities) + 1e-8)
    nearest = int(np.argmin(ball_scores))
    return nearest, float(ball_scores[nearest])


def _projected_case_geometry(det, view_data: np.ndarray, idx: int):
    norm_data = (view_data - det.mean) / det.std
    pca2 = PCA(n_components=2)
    emb = pca2.fit_transform(norm_data)
    centers = np.array([ball_obj.center for ball_obj in det.balls])
    center_emb = pca2.transform(centers)
    ball_idx, high_dim_score = _nearest_ball(det, view_data[idx])
    proj_members = pca2.transform(det.balls[ball_idx].data)
    proj_radius = np.percentile(np.linalg.norm(proj_members - center_emb[ball_idx], axis=1), 85)
    proj_radius = max(float(proj_radius), 0.02)
    proj_dist = float(np.linalg.norm(emb[idx] - center_emb[ball_idx]))
    return {
        "emb": emb,
        "center_emb": center_emb,
        "ball_idx": ball_idx,
        "proj_radius": proj_radius,
        "proj_dist": proj_dist,
        "proj_ratio": proj_dist / (proj_radius + 1e-8),
        "high_dim_score": high_dim_score,
    }


def _containing_ball_idx(det, sample_idx: int) -> int | None:
    for b_idx, ball_obj in enumerate(det.balls):
        if int(sample_idx) in {int(i) for i in ball_obj.indices}:
            return b_idx
    return None


def _shrink_to_nonoverlap(candidate: dict, chosen: list[dict], min_gap: float = 0.18) -> dict:
    adjusted = candidate.copy()
    max_radius = adjusted["radius"]
    for existing in chosen:
        distance = float(np.linalg.norm(adjusted["center"] - existing["center"]))
        max_radius = min(max_radius, distance - existing["radius"] - min_gap)
    if adjusted.get("highlighted"):
        target_radius = adjusted.get("target_radius", 0.02)
        adjusted["radius"] = max(target_radius, min(adjusted["radius"], max_radius))
    else:
        adjusted["radius"] = min(adjusted["radius"], max_radius)
    adjusted["radius"] = max(float(adjusted["radius"]), 0.02)
    return adjusted


def _representative_projected_balls(
    det,
    emb: np.ndarray,
    center_emb: np.ndarray,
    selected_ball_idx: int,
    highlighted_ball_idx: int | None = None,
    highlight_point: np.ndarray | None = None,
    max_circles: int = 3,
):
    """Return the scoring reference and, when distinct, membership ball."""
    candidates = []
    for b_idx, ball_obj in enumerate(det.balls):
        indices = [int(i) for i in ball_obj.indices]
        if not indices:
            continue
        proj_members = emb[indices]
        radius = np.percentile(np.linalg.norm(proj_members - center_emb[b_idx], axis=1), 85)
        radius = max(float(radius), 0.02)
        highlighted = highlighted_ball_idx is not None and b_idx == highlighted_ball_idx
        target_radius = 0.02
        if highlighted and highlight_point is not None:
            target_radius = float(np.linalg.norm(highlight_point - center_emb[b_idx])) * 1.10 + 0.04
            radius = max(radius, target_radius)
        candidates.append(
            {
                "idx": b_idx,
                "center": center_emb[b_idx],
                "radius": radius,
                "size": len(indices),
                "selected": b_idx == selected_ball_idx,
                "highlighted": highlighted,
                "target_radius": target_radius,
            }
        )

    selected = next((c for c in candidates if c["selected"]), None)
    if selected is None:
        return []

    chosen = [selected]
    highlighted = next((c for c in candidates if c["highlighted"] and not c["selected"]), None)
    if highlighted is not None and len(chosen) < max_circles:
        chosen.append(_shrink_to_nonoverlap(highlighted, chosen))

    return chosen


def _select_cases(labels: np.ndarray, anomaly_types: np.ndarray, final_scores: np.ndarray, views, view_scores, view_weights, detector):
    normal_candidates = np.where(labels == 0)[0]
    normal_idx = int(normal_candidates[np.argmin(final_scores[normal_candidates])])

    speed_candidates = np.where(anomaly_types == 2)[0]
    speed_idx = int(speed_candidates[np.argmax(final_scores[speed_candidates])])

    loop_candidates = np.where(anomaly_types == 1)[0]
    best_loop_idx = None
    best_loop_score = -np.inf
    for candidate in loop_candidates:
        dominant = _dominant_view(view_scores, view_weights, int(candidate))
        det = detector.detectors[dominant]
        geom = _projected_case_geometry(det, views[dominant], int(candidate))
        # Prefer loop cases that are visually outside or near the edge of the projected
        # reference ball, while still keeping a high final anomaly score.
        visual_term = min(geom["proj_ratio"], 2.2)
        score = 0.62 * float(final_scores[candidate]) + 0.38 * visual_term
        if geom["proj_ratio"] >= 0.92 and score > best_loop_score:
            best_loop_idx = int(candidate)
            best_loop_score = score
    if best_loop_idx is None:
        best_loop_idx = int(loop_candidates[np.argmax(final_scores[loop_candidates])])

    return [
        ("Normal trajectory", normal_idx),
        ("Speed anomaly", speed_idx),
        ("Loop anomaly", best_loop_idx),
    ]


def _dominant_view(view_scores: dict[str, np.ndarray], view_weights: dict[str, float], idx: int) -> str:
    contributions = {name: float(view_scores[name][idx] * view_weights[name]) for name in view_scores}
    return max(contributions, key=contributions.get)


def _format_feature(name: str) -> str:
    if name.startswith("shape_PC"):
        return name.replace("_", " ")
    return SHORT_FEATURE_NAMES.get(name, name.replace("_", " "))


def _local_distance_contributions(det, row: np.ndarray, ball_idx: int) -> np.ndarray:
    """Decompose the selected ball's weighted squared distance by feature."""
    normalized = (row - det.mean) / det.std
    delta = normalized - det.balls[ball_idx].center
    terms = np.asarray(det.local_weights[ball_idx], dtype=float) * np.square(delta)
    total = float(np.sum(terms))
    if not np.isfinite(total) or total <= 1e-12:
        return np.zeros_like(terms)
    return terms / total


def _clustered_point_indices(traj: np.ndarray, window: int = 6) -> np.ndarray:
    """Return the most locally compressed consecutive sample points."""
    if len(traj) <= window:
        return np.arange(len(traj))
    step_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    local_sums = np.array([
        step_lengths[start : start + window - 1].sum()
        for start in range(len(traj) - window + 1)
    ])
    start = int(np.argmin(local_sums))
    return np.arange(start, start + window)


def _relative_segment_speeds(traj: np.ndarray) -> np.ndarray:
    speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    lo, hi = np.percentile(speeds, [5, 95])
    if hi - lo <= 1e-8:
        return np.full_like(speeds, 0.5, dtype=float)
    return np.clip((speeds - lo) / (hi - lo), 0.0, 1.0)


def _relative_point_speeds(traj: np.ndarray) -> np.ndarray:
    seg = _relative_segment_speeds(traj)
    point = np.empty(len(traj), dtype=float)
    point[0] = seg[0]
    point[-1] = seg[-1]
    if len(traj) > 2:
        point[1:-1] = 0.5 * (seg[:-1] + seg[1:])
    return point


def _plot_speed_colored_trajectory(ax, traj: np.ndarray, case_name: str, lw: float = 2.05) -> None:
    cmap = CASE_SPEED_CMAPS[case_name]
    norm = Normalize(vmin=0.0, vmax=1.0)
    segments = np.stack([traj[:-1], traj[1:]], axis=1)
    seg_speed = _relative_segment_speeds(traj)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw, zorder=3)
    lc.set_array(seg_speed)
    ax.add_collection(lc)
    point_speed = _relative_point_speeds(traj)
    ax.scatter(
        traj[:, 0],
        traj[:, 1],
        s=12.5,
        c=point_speed,
        cmap=cmap,
        norm=norm,
        edgecolor="#FFFFFF",
        lw=0.45,
        alpha=0.96,
        zorder=4,
    )


def draw() -> None:
    n_normal = 1200
    n_anomaly_per_type = max(1, int(n_normal * 0.10 / (4 * 0.90)))
    trajs, labels, anomaly_types, _ = generate_synthetic_trajectories(
        n_normal=n_normal,
        n_anomaly_per_type=n_anomaly_per_type,
        seed=42,
    )
    spatial, kinematic, _, path_feat, _ = extract_all_features_v2(trajs)
    spatial_path = np.hstack([spatial, path_feat])

    detector = ThreeViewEWGBDetector(
        min_samples=8,
        purity_threshold=0.85,
        n_shape_dims=10,
        view_fusion="entropy",
        use_local_entropy=True,
    )
    detector.fit(spatial_path, kinematic, trajs)
    final_scores = _normalize_scores(detector.score(spatial_path, kinematic, trajs))
    views = _view_data(detector, spatial_path, kinematic, trajs)
    view_scores = {name: _normalize_scores(detector.detectors[name].score(data)) for name, data in views.items()}
    view_weights = detector.view_weights or {name: 1 / len(view_scores) for name in view_scores}
    cases = _select_cases(labels, anomaly_types, final_scores, views, view_scores, view_weights, detector)
    panel_letters = ["a", "b", "c"]

    fig = plt.figure(figsize=(7.95, 7.14))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        height_ratios=[1.10, 1.00, 0.82, 0.50],
        hspace=0.54,
        wspace=0.30,
        left=0.055,
        right=0.992,
        top=0.965,
        bottom=0.170,
    )

    top_axes = []
    for col, (case_name, idx) in enumerate(cases):
        color = CASE_COLORS[case_name]
        dominant = _dominant_view(view_scores, view_weights, idx)
        det = detector.detectors[dominant]
        geom = _projected_case_geometry(det, views[dominant], idx)
        ball_idx = int(geom["ball_idx"])
        ball = det.balls[ball_idx]

        ax_traj = fig.add_subplot(gs[0, col])
        top_axes.append(ax_traj)
        member_indices = [int(i) for i in ball.indices if labels[int(i)] == 0]
        if len(member_indices) == 0:
            member_indices = [int(i) for i in ball.indices]
        for member_idx in member_indices[:28]:
            member = trajs[member_idx]
            ax_traj.plot(member[:, 0], member[:, 1], color="#B8B8B8", lw=0.55, alpha=0.32, zorder=1)
        traj = trajs[idx]
        all_coords = np.vstack([*[trajs[i] for i in member_indices[:28]], traj])
        x_min, y_min = all_coords.min(axis=0)
        x_max, y_max = all_coords.max(axis=0)
        x_pad = max((x_max - x_min) * 0.06, 0.015)
        y_pad = max((y_max - y_min) * 0.10, 0.015)
        _plot_speed_colored_trajectory(ax_traj, traj, case_name)
        if case_name == "Speed anomaly":
            cluster = _clustered_point_indices(traj, window=6)
            cluster_xy = traj[cluster]
            cluster_speed = _relative_point_speeds(traj)[cluster]
            cmap = CASE_SPEED_CMAPS[case_name]
            norm = Normalize(vmin=0.0, vmax=1.0)
            span_x = max(float(cluster_xy[:, 0].max() - cluster_xy[:, 0].min()), 1e-4)
            span_y = max(float(cluster_xy[:, 1].max() - cluster_xy[:, 1].min()), 1e-4)
            full_x = max(float(x_max - x_min), 1e-4)
            full_y = max(float(y_max - y_min), 1e-4)
            ax_traj.plot(
                cluster_xy[:, 0],
                cluster_xy[:, 1],
                color="#7F2704",
                lw=2.85,
                alpha=0.88,
                solid_capstyle="round",
                zorder=5,
            )
            ax_traj.scatter(
                cluster_xy[:, 0],
                cluster_xy[:, 1],
                s=32,
                c=cluster_speed,
                cmap=cmap,
                norm=norm,
                edgecolor="#7F2704",
                lw=0.8,
                zorder=6,
            )
            slowest_xy = cluster_xy[int(np.argmin(cluster_speed))]
            ax_traj.scatter(
                slowest_xy[0],
                slowest_xy[1],
                s=64,
                color="#7F2704",
                edgecolor="#D55E00",
                lw=1.0,
                zorder=7,
            )
            ax_traj.add_patch(
                Ellipse(
                    xy=cluster_xy.mean(axis=0),
                    width=max(span_x * 3.2, full_x * 0.10),
                    height=max(span_y * 3.2, full_y * 0.14),
                    angle=0,
                    fill=False,
                    edgecolor="#D55E00",
                    lw=1.05,
                    linestyle=(0, (3, 2)),
                    alpha=0.95,
                    zorder=5,
                )
        )
        ax_traj.scatter(traj[0, 0], traj[0, 1], s=15, color="white", edgecolor=color, lw=0.8, zorder=4)
        ax_traj.scatter(traj[-1, 0], traj[-1, 1], s=18, color=color, marker="s", zorder=4)
        ax_traj.set_title(CASE_HEADERS.get(case_name, case_name), fontsize=10.0, pad=4, color="#222222")
        ax_traj.text(
            0.02,
            0.97,
            "1  Input trajectory",
            transform=ax_traj.transAxes,
            fontsize=7.4,
            fontweight="bold",
            color="#444444",
            ha="left",
            va="top",
        )
        ax_traj.text(
            -0.08,
            1.08,
            panel_letters[col],
            transform=ax_traj.transAxes,
            fontsize=10.0,
            fontweight="bold",
            color=color,
            ha="left",
            va="bottom",
        )
        ax_traj.set_xlim(x_min - x_pad, x_max + x_pad)
        ax_traj.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        ax_traj.spines["top"].set_visible(False)
        ax_traj.spines["right"].set_visible(False)
        ax_traj.set_xlabel("$x$", fontsize=8.0, labelpad=1)
        if col == 0:
            ax_traj.set_ylabel("$y$\ncoordinate", fontsize=8.5)

        ax_ball = fig.add_subplot(gs[1, col])
        norm_data = (views[dominant] - det.mean) / det.std
        emb = geom["emb"]
        center_emb = geom["center_emb"]
        member_mask = np.zeros(len(labels), dtype=bool)
        member_mask[[int(i) for i in ball.indices]] = True
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        ax_ball.scatter(
            emb[normal_mask, 0],
            emb[normal_mask, 1],
            s=5,
            color="#D0D0D0",
            alpha=0.28,
            lw=0,
            zorder=1,
        )
        ax_ball.scatter(
            emb[anomaly_mask, 0],
            emb[anomaly_mask, 1],
            s=7,
            color="#F2C8B8",
            alpha=0.32,
            lw=0,
            zorder=1,
        )
        ax_ball.scatter(
            emb[member_mask, 0],
            emb[member_mask, 1],
            s=9,
            color=color,
            alpha=0.34,
            lw=0,
            zorder=2,
        )
        anomaly_ball_idx = _containing_ball_idx(det, idx) if labels[idx] == 1 else None
        representative_balls = _representative_projected_balls(
            det,
            emb,
            center_emb,
            ball_idx,
            highlighted_ball_idx=anomaly_ball_idx,
            highlight_point=emb[idx] if anomaly_ball_idx is not None else None,
            max_circles=1 if labels[idx] == 0 else 2,
        )
        for info in sorted(
            representative_balls,
            key=lambda item: 2 if item["highlighted"] and not item["selected"] else (1 if item["selected"] else 0),
        ):
            linestyle = "solid"
            if info["highlighted"] and not info["selected"]:
                edge = color
                face = color
                lw = 2.05
                face_alpha = 0.035
                edge_alpha = 0.98
                linestyle = (0, (3.2, 2.0))
                zorder = 5
            elif info["selected"]:
                edge = color
                face = color
                lw = 2.20
                face_alpha = 0.12
                edge_alpha = 0.98
                zorder = 4
            else:
                edge = "#4A4A4A"
                face = "#DADADA"
                lw = 1.75
                face_alpha = 0.035
                edge_alpha = 0.95
                zorder = 3
            ax_ball.add_patch(
                Circle(
                    info["center"],
                    info["radius"],
                    facecolor=to_rgba(face, face_alpha),
                    edgecolor=to_rgba(edge, edge_alpha),
                    lw=lw,
                    linestyle=linestyle,
                    zorder=zorder,
                )
            )
        ax_ball.scatter(emb[idx, 0], emb[idx, 1], s=42, marker="*", color=color, edgecolor="black", lw=0.45, zorder=5)
        ax_ball.set_title(
            f"2  Dominant-view reference ball: {VIEW_LABELS[dominant]}",
            fontsize=8.8,
            pad=3,
        )
        ax_ball.set_xticks([])
        ax_ball.set_yticks([])
        ax_ball.spines["top"].set_visible(False)
        ax_ball.spines["right"].set_visible(False)
        if col == 0:
            ax_ball.set_ylabel("Projected\nview space", fontsize=8.5)

        ax_weight = fig.add_subplot(gs[2, col])
        names = FEATURE_NAMES[dominant]
        contributions_by_feature = _local_distance_contributions(
            det, views[dominant][idx], ball_idx
        )
        top_k = min(5, len(contributions_by_feature))
        order = np.argsort(contributions_by_feature)[-top_k:][::-1]
        y = np.arange(top_k)
        contribution_pct = 100.0 * contributions_by_feature[order]
        bars = ax_weight.barh(
            y,
            contribution_pct,
            color=color,
            alpha=0.78,
            height=0.58,
            label="weighted distance share",
        )
        ax_weight.bar_label(
            bars,
            labels=[f"{value:.0f}%" for value in contribution_pct],
            padding=2,
            fontsize=6.2,
            color="#333333",
        )
        ax_weight.set_yticks(y)
        ax_weight.set_yticklabels([_format_feature(names[i]) for i in order], fontsize=7.3)
        ax_weight.invert_yaxis()
        max_pct = float(np.max(contribution_pct)) if len(contribution_pct) else 0.0
        ax_weight.set_xlim(0, max(35.0, max_pct * 1.22))
        ax_weight.tick_params(axis="x", labelsize=7.2)
        ax_weight.grid(axis="x", ls=":", lw=0.45, color="#999999", alpha=0.35)
        ax_weight.set_xlabel("share of weighted distance (%)", fontsize=7.0, labelpad=2)
        ax_weight.set_title("3  Feature contributions", fontsize=9.0, pad=3)
        ax_weight.spines["top"].set_visible(False)
        ax_weight.spines["right"].set_visible(False)
        if col == 0:
            ax_weight.set_ylabel("Top contributors", fontsize=8.5)

        ax_score = fig.add_subplot(gs[3, col])
        left = 0.0
        contributions = {name: float(view_scores[name][idx] * view_weights[name]) for name in view_scores}
        total = sum(contributions.values()) + 1e-12
        for view_name in ["spatial_path", "kinematic", "trajectory_shape"]:
            value = contributions[view_name]
            ax_score.barh(
                [0],
                [value],
                left=left,
                height=0.45,
                color=VIEW_COLORS[view_name],
                alpha=0.88,
                edgecolor="white",
                lw=0.35,
            )
            left += value
        ax_score.set_xlim(0, max(0.12, total * 1.10))
        ax_score.set_yticks([])
        ax_score.tick_params(axis="x", labelsize=7.2)
        ax_score.set_xlabel(
            rf"$\widetilde{{s}}=\sum_m\alpha_m\widehat{{s}}^{{(m)}}={total:.2f}"
            rf"\;\rightarrow\;s={final_scores[idx]:.2f}$",
            fontsize=7.0,
            labelpad=3,
        )
        ax_score.set_title("4  Weighted view contributions", fontsize=9.0, pad=3)
        ax_score.spines["top"].set_visible(False)
        ax_score.spines["right"].set_visible(False)
        ax_score.spines["left"].set_visible(False)
        if col == 0:
            ax_score.set_ylabel("Weighted\nscores", fontsize=8.5)

    for left_ax, right_ax in zip(top_axes[:-1], top_axes[1:]):
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        x_sep = (left_pos.x1 + right_pos.x0) / 2
        fig.add_artist(
            Line2D(
                [x_sep, x_sep],
                [0.155, 0.965],
                transform=fig.transFigure,
                color="#D9D9D9",
                lw=0.55,
                linestyle=(0, (2.2, 2.2)),
                zorder=0,
            )
        )

    speed_cax = fig.add_axes([0.090, 0.092, 0.145, 0.009])
    speed_grad = np.linspace(0, 1, 256)[np.newaxis, :]
    speed_cmap = LinearSegmentedColormap.from_list("speed_legend", ["#7F2704", "#FEE6CE"])
    speed_cax.imshow(speed_grad, aspect="auto", cmap=speed_cmap)
    speed_cax.set_xticks([0, 255])
    speed_cax.set_xticklabels(["low speed", "high speed"], fontsize=6.7)
    speed_cax.set_yticks([])
    for spine in speed_cax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=VIEW_COLORS[name], lw=5, label=VIEW_LABELS[name])
        for name in ["spatial_path", "kinematic", "trajectory_shape"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.60, 0.072),
        ncol=3,
        frameon=False,
        fontsize=8.0,
        handlelength=1.4,
        columnspacing=1.5,
    )

    for ext in ("pdf", "png", "svg", "tiff"):
        out = PAPER_DIR / f"mechanism_cases.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 700
        fig.savefig(out, **kwargs)
        print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    draw()
