# Outline proposal

## Project scan summary

- Project archetype candidate: mixed risk-model and empirical benchmark.
- Supporting evidence from files: `rv_estimators.py` defines variance estimators;
  `build_features.py` enforces feature timing; `har.py`, `garch.py`, `linear.py`,
  and `xgboost_model.py` implement five forecasters; `train_evaluate.py` runs an
  expanding-window comparison with QLIKE, MSE, diagnostics, and Diebold-Mariano
  tests.
- Reproduced defect: the former five-calendar-year burn-in extended beyond the
  2022-11-02 to 2024-12-20 feature history and silently wrote empty outputs.
- Resolution: the portable default now uses one calendar year, which yields 286
  forecast origins per model and horizon. Infeasible windows raise a dated error.

## Blueprint selection

- Selected blueprint: mixed.
- Why this blueprint fits this project: the article must explain realised
  variance and conditional-variance models, then assess them through a strict
  chronological benchmark rather than a trading backtest.
- Planned section order:
  1. The clock is part of the benchmark.
  2. From minute returns to forecast targets.
  3. Feature availability and no-lookahead alignment.
  4. Five model families and their equations.
  5. The repaired expanding-window protocol.
  6. QLIKE, MSE, and Diebold-Mariano definitions.
  7. Corrected default results and forecast-path evidence.
  8. Limits, interpretation, and primary references.

## Planned equations

1. Intraday log return and realised variance.
   - Purpose: derive the daily forecast target from minute closes.
   - Symbols: minute price $P_{t,i}$, return $r_{t,i}$, count $M_t$, realised
     variance $RV_t$.
   - Delimiter: display.
2. One-day and five-day targets.
   - Purpose: show exactly which future observations each feature row predicts.
   - Symbols: feature date $t$, horizon $h$, target $y_{t,h}$.
   - Delimiter: display.
3. HAR-RV daily, weekly, and monthly regressors.
   - Purpose: explain multi-horizon volatility persistence.
   - Symbols: lagged averages $x_{d,t}$, $x_{w,t}$, $x_{m,t}$ and coefficients
     $\beta$.
   - Delimiter: display.
4. GARCH(1,1) return and conditional-variance recursion.
   - Purpose: contrast a return-driven variance model with HAR-RV.
   - Symbols: return $r_t$, shock $\varepsilon_t$, conditional standard deviation
     $\sigma_t$, and parameters $\omega$, $\alpha$, $\beta$.
   - Delimiter: display.
5. Ridge and Lasso objectives in log-variance space.
   - Purpose: explain positivity and the difference between L2 and L1 penalties.
   - Symbols: log target $z_t$, feature vector $\mathbf{x}_t$, coefficients
     $\boldsymbol\theta$, penalty $\lambda$.
   - Delimiter: display.
6. Additive tree ensemble.
   - Purpose: state the XGBoost prediction structure without pretending trees
     supply a closed-form volatility model.
   - Symbols: trees $f_m$, learning rate $\eta$, prediction $F_M$.
   - Delimiter: display.
7. MSE and QLIKE.
   - Purpose: derive the two rankings and explain why they disagree.
   - Symbols: realised variance $y_t$, forecast $\widehat y_t$, sample size $n$.
   - Delimiter: display.
8. Diebold-Mariano loss differential.
   - Purpose: interpret the pairwise test direction and caveats.
   - Symbols: model losses $L_{a,t}$ and $L_{b,t}$, difference $d_t$.
   - Delimiter: display.

## Planned code excerpts

1. File: `src/volcast/features/rv_estimators.py`.
   - Function/block: shifted HAR rolling mean.
   - Why include this excerpt: proves the feature at date $t$ excludes $RV_t$.
2. File: `src/volcast/models/garch.py`.
   - Function/block: variance recursion.
   - Why include this excerpt: connects the GARCH equation to implementation.
3. File: `src/volcast/models/linear.py`.
   - Function/block: inverse log-target transform.
   - Why include this excerpt: explains positive forecasts without routine floor
     clipping.
4. File: `src/volcast/evaluation/train_evaluate.py`.
   - Function/block: infeasible-window exception.
   - Why include this excerpt: documents the corrected evidence contract.

## Planned technical graphs

1. Graph type: realised-volatility time series and 22-day mean.
   - Source: generated from default stage-2 output.
   - Expected takeaway: volatility clustering makes chronological evaluation
     necessary.
2. Graph type: QLIKE difference from HAR-RV by model and horizon.
   - Source: generated from corrected default stage-4 scores.
   - Expected takeaway: Lasso and Ridge improve QLIKE while HAR-RV wins MSE.
3. Graph type: realised versus forecast one-day annualised volatility.
   - Source: generated from corrected default forecasts.
   - Expected takeaway: both HAR-RV and Lasso smooth short realised spikes, which
     aggregate rankings alone do not show.

## Risks, gaps, and assumptions

- Data gaps: one asset, roughly two years, and no independent audit of the
  packaged market-data source.
- Assumptions: 252 trading days per year; option and VIX observations are usable
  one business day later; bounded three-day forward-fill remains economically
  meaningful.
- Statistical cautions: overlapping five-day targets create serial dependence;
  pairwise tests are numerous and unadjusted; the one-year burn-in is a portable
  design choice rather than an estimated optimum.
- Validation checks: full tests and Ruff, default pipeline, explicit five-year
  failure reproduction, chart regeneration, both blog validators, frontmatter
  and image checks, bilingual protected-block comparison, and primary-reference
  link verification.
- Deployment: canonical files stay in `vol-forecast-benchmarks/blog/`. The user
  explicitly prohibited website changes, Hugo publication, and website commits.

## Outline review

The revised structure gives the corrected default evidence precedence over the
historical defect. It covers context, methodology, evidence, and limitations.
Every planned equation maps to executable code or a cited primary paper, and the
three graphs answer different questions rather than repeating the score table.
