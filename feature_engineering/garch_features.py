import numpy as np
from arch import arch_model


def add_garch_features(df, p=1, q=1):
    """
    Adds GARCH(1,1) conditional volatility estimates as a feature.

    Expected columns:
        - log_return
    """

    df = df.copy()

    # Scale returns to percentages (recommended for arch package)
    returns = df["log_return"] * 100

    # Remove missing values for fitting
    valid_returns = returns.dropna()

    # Need enough observations
    if len(valid_returns) < 100:
        df["garch_forecast"] = np.nan
        return df

    # Fit GARCH model once
    model = arch_model(
        valid_returns,
        vol="Garch",
        p=p,
        q=q,
        mean="Zero"
    )

    result = model.fit(disp="off")

    # Convert back from percentages
    conditional_vol = result.conditional_volatility / 100

    # Create output column
    df["garch_forecast"] = np.nan

    # Align with original DataFrame indices
    df.loc[conditional_vol.index, "garch_forecast"] = conditional_vol.values

    return df