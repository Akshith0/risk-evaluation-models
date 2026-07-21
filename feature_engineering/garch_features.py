import numpy as np
from arch import arch_model


def add_garch_features(df, p=1, q=1, lookback=252, horizon=10):
    df = df.copy()
    df["garch_forecast"] = np.nan

    returns = df["log_return"].astype(float) * 100

    for i in range(lookback, len(returns) - horizon):
        train_returns = returns.iloc[i - lookback:i].dropna()

        if len(train_returns) < lookback:
            continue

        try:
            model = arch_model(
                train_returns,
                vol="Garch",
                p=p,
                q=q,
                mean="Zero"
            )

            result = model.fit(disp="off")

            forecast = result.forecast(horizon=horizon)

            variances = forecast.variance.iloc[-1].values

            ten_day_vol = np.sqrt(np.sum(variances)) / 100

            annualized_vol = ten_day_vol * np.sqrt(252 / horizon)

            df.loc[df.index[i], "garch_forecast"] = annualized_vol

        except Exception:
            continue

    return df