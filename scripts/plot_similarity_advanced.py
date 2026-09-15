from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS


ROOT = Path("figures/penmanshiel_intro")
FIGURE_DIR = ROOT / "02_similarity"
DATA_DIR = FIGURE_DIR / "data"
PNG_DIR = FIGURE_DIR / "png"


def _load_similarity() -> tuple[list[str], np.ndarray]:
    table = pd.read_csv(DATA_DIR / "figure2_pearson.csv", index_col=0)
    turbines = list(table.index.astype(str))
    matrix = table.to_numpy(dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 1.0)
    return turbines, matrix


def _load_positions(turbines: list[str]) -> dict[str, tuple[float, float]]:
    static_path = ROOT / "shared_static_coordinates.csv"
    if not static_path.exists():
        return {}
    static = pd.read_csv(static_path).set_index("turbine_id")
    positions = {}
    for turbine in turbines:
        if turbine in static.index:
            x = static.loc[turbine, "x"]
            y = static.loc[turbine, "y"]
            if np.isfinite(x) and np.isfinite(y):
                positions[turbine] = (float(x), float(y))
    return positions


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _save(fig: plt.Figure, filename: str) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_nodes(
    axis: plt.Axes,
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    matrix: np.ndarray,
    turbines: list[str],
) -> None:
    node_values = {
        turbine: float(np.mean(np.delete(matrix[index], index)))
        for index, turbine in enumerate(turbines)
    }
    node_colors = [node_values[node] for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=440,
        node_color=node_colors,
        cmap="viridis",
        vmin=min(node_values.values()),
        vmax=max(node_values.values()),
        edgecolors="white",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=axis,
        font_size=8,
        font_weight="bold",
        font_color="white",
    )


def plot_similarity_network(turbines: list[str], matrix: np.ndarray) -> None:
    positions = _load_positions(turbines)
    graph = nx.Graph()
    graph.add_nodes_from(turbines)
    values = _off_diagonal(matrix)
    threshold = float(np.quantile(values, 0.75))
    for i, source in enumerate(turbines):
        for j, target in enumerate(turbines):
            if j <= i:
                continue
            weight = float(matrix[i, j])
            if weight >= threshold:
                graph.add_edge(source, target, weight=weight)
    if len(positions) != len(turbines):
        positions = nx.spring_layout(graph, seed=2026, weight="weight")

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    edge_weights = np.array([graph[u][v]["weight"] for u, v in graph.edges])
    widths = 0.8 + 5.0 * (edge_weights - edge_weights.min()) / max(
        float(edge_weights.max() - edge_weights.min()), 1e-12
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=axis,
        edge_color=edge_weights,
        edge_cmap=plt.cm.plasma,
        edge_vmin=threshold,
        edge_vmax=1.0,
        width=widths,
        alpha=0.78,
    )
    _draw_nodes(axis, graph, positions, matrix, turbines)
    axis.set_title("High-similarity turbine network", fontsize=13, pad=10)
    axis.set_xlabel("East-west coordinate (m)")
    axis.set_ylabel("North-south coordinate (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.text(
        0.01,
        0.02,
        f"Edges retain top 25% pairwise Pearson r (r >= {threshold:.3f})",
        transform=axis.transAxes,
        fontsize=8,
        color="0.25",
    )
    for spine in axis.spines.values():
        spine.set_color("0.35")
    _save(fig, "figure2_advanced_similarity_network.png")


def plot_mst_backbone(turbines: list[str], matrix: np.ndarray) -> None:
    positions = _load_positions(turbines)
    graph = nx.Graph()
    for i, source in enumerate(turbines):
        for j, target in enumerate(turbines):
            if j <= i:
                continue
            graph.add_edge(source, target, distance=float(1.0 - matrix[i, j]), similarity=float(matrix[i, j]))
    tree = nx.minimum_spanning_tree(graph, weight="distance")
    if len(positions) != len(turbines):
        positions = nx.spring_layout(tree, seed=2026, weight="similarity")

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    similarities = np.array([tree[u][v]["similarity"] for u, v in tree.edges])
    widths = 1.2 + 5.5 * (similarities - similarities.min()) / max(
        float(similarities.max() - similarities.min()), 1e-12
    )
    nx.draw_networkx_edges(
        tree,
        positions,
        ax=axis,
        edge_color=similarities,
        edge_cmap=plt.cm.cividis,
        edge_vmin=float(similarities.min()),
        edge_vmax=float(similarities.max()),
        width=widths,
        alpha=0.88,
    )
    _draw_nodes(axis, tree, positions, matrix, turbines)
    axis.set_title("Minimum-spanning similarity backbone", fontsize=13, pad=10)
    axis.set_xlabel("East-west coordinate (m)")
    axis.set_ylabel("North-south coordinate (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.text(
        0.01,
        0.02,
        "Tree minimizes total distance, where distance = 1 - Pearson r",
        transform=axis.transAxes,
        fontsize=8,
        color="0.25",
    )
    for spine in axis.spines.values():
        spine.set_color("0.35")
    _save(fig, "figure2_advanced_mst_backbone.png")


def plot_mds_embedding(turbines: list[str], matrix: np.ndarray) -> None:
    distance = np.maximum(1.0 - matrix, 0.0)
    np.fill_diagonal(distance, 0.0)
    embedding = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=2026,
        normalized_stress="auto",
        n_init=8,
    ).fit_transform(distance)
    mean_similarity = np.array(
        [np.mean(np.delete(matrix[index], index)) for index in range(len(turbines))]
    )

    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=150,
        c=mean_similarity,
        cmap="viridis",
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
    )
    for turbine, x, y in zip(turbines, embedding[:, 0], embedding[:, 1]):
        axis.annotate(turbine, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.axhline(0, color="0.82", linewidth=0.8, zorder=1)
    axis.axvline(0, color="0.82", linewidth=0.8, zorder=1)
    axis.set_title("Similarity geometry by metric MDS", fontsize=13, pad=10)
    axis.set_xlabel("MDS 1")
    axis.set_ylabel("MDS 2")
    axis.grid(alpha=0.16, linewidth=0.6)
    colorbar = fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean Pearson r")
    _save(fig, "figure2_advanced_mds_embedding.png")


def plot_clustered_heatmap(turbines: list[str], matrix: np.ndarray) -> None:
    distance = np.maximum(1.0 - matrix, 0.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    tree = optimal_leaf_ordering(tree, condensed)
    order = dendrogram(tree, no_plot=True)["leaves"]
    ordered = matrix[np.ix_(order, order)]
    ordered_turbines = [turbines[index] for index in order]
    vmin = max(0.0, np.floor(float(np.min(_off_diagonal(matrix))) * 100) / 100)

    fig = plt.figure(figsize=(6.8, 7.2))
    fig.suptitle(
        "Hierarchically ordered similarity matrix",
        x=0.5,
        y=0.965,
        ha="center",
        fontsize=12,
    )
    grid = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[0.95, 4.6, 0.22],
        height_ratios=[0.9, 4.6, 0.22],
        wspace=0.20,
        hspace=0.04,
        left=0.07,
        right=0.90,
        top=0.88,
        bottom=0.12,
    )
    top_axis = fig.add_subplot(grid[0, 1])
    left_axis = fig.add_subplot(grid[1, 0])
    heat_axis = fig.add_subplot(grid[1, 1])
    colorbar_axis = fig.add_subplot(grid[1, 2])

    dendrogram(
        tree,
        ax=top_axis,
        color_threshold=0,
        above_threshold_color="0.25",
        no_labels=True,
    )
    top_axis.axis("off")
    dendrogram(
        tree,
        ax=left_axis,
        orientation="left",
        color_threshold=0,
        above_threshold_color="0.25",
        no_labels=True,
    )
    left_axis.axis("off")

    image = heat_axis.imshow(
        ordered,
        vmin=vmin,
        vmax=1.0,
        cmap="YlGnBu",
        interpolation="nearest",
    )
    heat_axis.set_xticks(range(len(ordered_turbines)), ordered_turbines, rotation=45, ha="right")
    heat_axis.set_yticks(range(len(ordered_turbines)), ordered_turbines)
    heat_axis.tick_params(axis="both", labelsize=8, length=0)
    heat_axis.set_xticks(np.arange(-0.5, len(ordered_turbines), 1), minor=True)
    heat_axis.set_yticks(np.arange(-0.5, len(ordered_turbines), 1), minor=True)
    heat_axis.grid(which="minor", color="white", linewidth=0.35)
    heat_axis.tick_params(which="minor", length=0)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Pearson r", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    _save(fig, "figure2_advanced_clustered_similarity.png")


def plot_similarity_distribution(turbines: list[str], matrix: np.ndarray) -> None:
    observed = _off_diagonal(matrix)
    cache = np.load(ROOT / "aligned_penmanshiel_arrays.npz", allow_pickle=False)
    speed = cache["speed"].astype(np.float64)
    speed = speed[:, : len(turbines)]
    rng = np.random.default_rng(2026)
    null_values = []
    for _ in range(120):
        shifted = speed.copy()
        for col in range(shifted.shape[1]):
            shifted[:, col] = np.roll(shifted[:, col], int(rng.integers(1, shifted.shape[0] - 1)))
        null_corr = np.corrcoef(shifted, rowvar=False)
        null_values.append(_off_diagonal(null_corr))
    null_values = np.concatenate(null_values)

    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    bins = np.linspace(min(float(null_values.min()), float(observed.min())), 1.0, 44)
    axis.hist(
        null_values,
        bins=bins,
        density=True,
        color="#9aa0a6",
        alpha=0.45,
        label="Circular-shift baseline",
    )
    axis.hist(
        observed,
        bins=bins,
        density=True,
        color="#006d9c",
        alpha=0.76,
        label="Observed turbine pairs",
    )
    axis.axvline(float(np.mean(observed)), color="#004b6b", linewidth=1.8)
    axis.axvline(float(np.mean(null_values)), color="0.35", linewidth=1.4, linestyle="--")
    axis.set_title("Observed similarity against a temporal-null baseline", fontsize=12, pad=8)
    axis.set_xlabel("Pairwise Pearson r")
    axis.set_ylabel("Density")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    _save(fig, "figure2_advanced_similarity_distribution.png")


def main() -> None:
    turbines, matrix = _load_similarity()
    plot_clustered_heatmap(turbines, matrix)
    plot_similarity_network(turbines, matrix)
    plot_mds_embedding(turbines, matrix)
    plot_mst_backbone(turbines, matrix)
    plot_similarity_distribution(turbines, matrix)
    print(f"advanced_similarity_png_dir={PNG_DIR.resolve()}")


if __name__ == "__main__":
    main()
