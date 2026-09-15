# Introduction Figure Plan

## Core Argument

The Penmanshiel data motivate a common-field and residual formulation:
multi-turbine wind-speed channels are strongly synchronized, while the
turbine-specific information is concentrated in a smaller residual component.
Wind direction is circular and should not be treated as an ordinary scalar.

The introduction should use two main figures. Additional visualizations belong
in the method and experimental sections.

## Figure 1: Evidence of a Shared Farm-Wide Dynamics

Use a four-panel figure:

1. Aligned wind-speed traces from several turbines.
2. Pairwise Pearson-correlation heatmap.
3. First principal-component and cumulative-variance curves.
4. Common field together with turbine residuals.

The figure should establish that the dataset is neither a collection of
independent univariate series nor a perfectly identical signal.

## Figure 2: Why Phase-Net Decomposes the Forecasting Problem

Compare two pipelines:

```text
Generic multivariate model:
all turbine channels -> one large temporal mixer

Phase-Net:
all turbine channels -> common-field branch
                   -> turbine-residual branch
                   -> speed/direction reconstruction
```

Use arrows and short labels rather than a dense block diagram. This figure
should appear at the end of the introduction.

## Strong Supporting Visualizations

### Correlation and distance structure

- Pearson-correlation heatmap for level series.
- Spearman-correlation heatmap as a robustness check.
- Cosine-similarity heatmap after centering and normalizing each turbine.
- Distance correlation or mutual-information matrix if sample size permits.

Cosine similarity alone should not be used as the main evidence because wind
speed levels are positive and can make similarity appear artificially high.

### Low-dimensional structure

- PCA score plot of turbine trajectories.
- Cumulative explained-variance curve.
- 2D t-SNE or UMAP of fixed-length turbine windows.
- Color the embedding by turbine identity and by wind-speed regime.

t-SNE/UMAP is useful for visualization but not for proving numerical
dependence. PCA and correlation statistics should carry the main claim.

### Common-field and residual structure

- Distribution of raw and residual standard deviations by turbine.
- Boxplots of pairwise correlations before and after common-field removal.
- Singular-value spectrum of the turbine-by-time speed matrix.
- Explained-variance curve versus number of common factors.
- Heatmap of residual correlations.

These plots directly justify using a low-dimensional common branch while
retaining a turbine-specific residual branch.

### Wind-direction geometry

- Unit-circle plot of direction values in `(cos(theta), sin(theta))`.
- Example crossing from `359` degrees to `1` degree.
- Scalar-angle error versus circular-angle error.
- Direction-sector histogram.

This figure supports the circular direction representation independently from
the common-field argument.

## Recommended Main-Text Order

1. Figure 1: shared farm-wide dynamics.
2. Figure 2: motivation and Phase-Net overview.
3. Figure 3: detailed phase-aware architecture.
4. Figure 4: circular wind-direction representation.
5. Main accuracy and efficiency results.

The remaining correlation variants, t-SNE/UMAP plots, residual distributions,
and sensitivity analyses should be placed in supplementary material unless
they reveal a clear additional result.

## Claims to Avoid

- Do not claim all turbines are identical.
- Do not use t-SNE as a statistical significance test.
- Do not claim geographical information is used in the current final Shanxi
  configuration.
- Do not call a screened candidate period a strict physical period without a
  separate periodicity analysis.
