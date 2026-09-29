# Architecture

Rendered copy: [`architecture_forecasting.png`](../../docs/figures/architecture_forecasting.png) (source below).

```mermaid
flowchart LR
    A[Versioned onboarding outputs] --> B[Schema key hash and manifest verification]
    B --> B2[Fault cutoff and exact-change verification]
    B2 --> C[Time-aware feature builder: lags, calendar, lagged site weather]
    C --> D[Chronological split from reference]
    D --> E1[Seasonal naive]
    D --> E2[Ridge regression]
    D --> E3[Random forest]
    D --> E4[Gradient boosting]
    E1 --> F[Common-key alignment]
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G[MASE MAE RMSE wMAPE R2]
    G --> H[Bootstrap intervals paired differences and subgroup analysis]
    H --> I[Machine-readable results]
    I --> J[Generated report and run manifest]
```

`verify-inputs` fails before model fitting if producer files do not match their manifests, condition keys differ, a changed corrupted value is undocumented, or a fault violates its declared cutoff. The experiment then enforces that actual faults are training-only. Test targets are joined exclusively from reference data, and finite predictions are re-aligned onto one common key set before any metric is computed.

`verify-results` checks the required artifacts, result hashes, unique prediction keys, condition/model coverage, shared targets, common-key alignment, paired output, and four subgroup dimensions. The pipeline is stateless between runs. Reproducibility comes from versioned public data, hashes, code, configurations, seeds, and machine-readable outputs.
