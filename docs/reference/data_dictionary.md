# Data Dictionary

## Raw Inputs (`data/raw`)

### `SPY_minutes.parquet`

| Column | Type | Meaning |
|---|---|---|
| `symbol` | string | Asset ticker (`SPY`) |
| `ts` | timestamp | Minute-bar timestamp (New York timezone) |
| `open` | float | Minute open price |
| `high` | float | Minute high price |
| `low` | float | Minute low price |
| `close` | float | Minute close price |
| `volume` | float | Minute traded volume |

### `SPY_options.parquet/part-*.parquet`

| Column | Type | Meaning |
|---|---|---|
| `symbol` | string | Underlying ticker (`SPY`) |
| `trade_date` | timestamp | Option chain observation date |
| `strike_price` | float | Option strike |
| `expiry_date` | timestamp | Contract expiry date |
| `option_type` | string | `c` call or `p` put |
| `bid` | float | Bid quote |
| `ask` | float | Ask quote |
| `bid_iv` | float | Bid implied volatility |
| `ask_iv` | float | Ask implied volatility |
| `delta` | float | Option delta |
| `open_interest` | integer | Open interest |
| `volume` | integer | Contract volume |

### `VIX_daily.parquet`

| Column | Type | Meaning |
|---|---|---|
| `date` | timestamp | Trading date |
| `vix` | float | Daily VIX close |

## Stage-2 Output (`data/processed/*_rv.parquet`)

| Column | Meaning |
|---|---|
| `date` | Trading date |
| `rv_cc` | Close-to-close realised variance |
| `rv_parkinson` | Parkinson variance estimator |
| `rv_gk` | Garman-Klass variance estimator |
| `n_minutes` | Number of minute bars kept that day |
| `daily_open`, `daily_high`, `daily_low`, `daily_close` | Daily OHLC summary from minute bars |
| `daily_returns` | Daily log return for GARCH input |
| `rv_cc_d`, `rv_cc_w`, `rv_cc_m` | HAR lags of close-to-close RV |
| `rv_pk_d`, `rv_pk_w`, `rv_pk_m` | HAR lags of Parkinson RV |
| `rv_gk_d`, `rv_gk_w`, `rv_gk_m` | HAR lags of Garman-Klass RV |
| `rv_1d_ahead` | One-day-ahead RV target |
| `rv_5d_ahead` | Five-day-ahead mean RV target |

## Stage-3 Output (`data/processed/*_features.parquet`)

| Column | Meaning |
|---|---|
| `date` | Model row date |
| `daily_returns` | Daily log return (for GARCH model path) |
| RV lag columns | Historical realised-variance features |
| `atm_iv` | At-the-money implied volatility (lagged) |
| `iv_skew` | Put-call IV skew (lagged) |
| `iv_term_slope` | Term-structure IV slope (lagged) |
| `iv_rv_spread` | ATM IV minus annualized weekly RV volatility |
| `vix` | Lagged daily VIX |
| `rv_1d_ahead`, `rv_5d_ahead` | Forecast targets |

## Stage-4 Outputs (`outputs`)

### `forecasts.parquet`

| Column | Meaning |
|---|---|
| `date` | Forecast date |
| `symbol` | Asset ticker |
| `horizon` | Forecast horizon in trading days |
| `model` | Model name |
| `y_true` | Realised variance target |
| `y_pred` | Predicted variance |

### `scores.parquet`

| Column | Meaning |
|---|---|
| `symbol` | Asset ticker |
| `horizon` | Forecast horizon |
| `model` | Model name |
| `qlike` | Mean QLIKE loss |
| `mse` | Mean squared error |
| `n_obs` | Number of scored observations |

### `dm_tests.parquet`

| Column | Meaning |
|---|---|
| `symbol` | Asset ticker |
| `horizon` | Forecast horizon |
| `model_a`, `model_b` | Compared models |
| `t_stat` | Diebold-Mariano t-statistic |
| `p_value` | Two-sided p-value |

### `model_diagnostics.parquet`

| Column | Meaning |
|---|---|
| `symbol`, `horizon`, `model` | Group identifier |
| `n_obs` | Valid forecast count |
| `floor_hit_count`, `floor_hit_rate` | Frequency of floor-clipped predictions |
| `min_prediction`, `median_prediction`, `max_prediction` | Prediction range summary |
| `qlike_median`, `qlike_p95` | Central and tail QLIKE diagnostics |
