import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def plot_series(df, *cols, date_col="Date",
                start="2015-11-01", end="2025-08-31"):
    """
    Plot 1–n columns over time.

    - Uses date_col if it exists, otherwise assumes DatetimeIndex.
    - Filters between `start` and `end`.
    - Y-values are used as-is (already in percent).
    """

    # allow passing a list: plot_series(df, ["UR", "male_ur"])
    if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
        cols = list(cols[0])
    else:
        cols = list(cols)

    # Make a working copy
    data = df.copy()

    # Case 1: Date is a column
    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col])
        mask = (data[date_col] >= start) & (data[date_col] <= end)
        data = data.loc[mask, [date_col] + cols]
        x = data[date_col]

    # Case 2: Date is already the index
    else:
        # assume DatetimeIndex
        data = data.loc[start:end, cols]
        x = data.index

    if data.empty:
        print("⚠️ No data in the selected date range or columns. "
              "Check your dates and column names.")
        return

    plt.figure(figsize=(12, 5))

    for col in cols:
        plt.plot(x, data[col], label=col, linewidth=2)

    # Title
    if len(cols) == 1:
        title_cols = cols[0]
    else:
        title_cols = ", ".join(cols)
    plt.title(f"{title_cols} ({start[:7]} – {end[:7]})", fontsize=14)

    plt.xlabel("Date")
    plt.ylabel("Percent")
    plt.grid(True, linestyle="--", alpha=0.5)
    if len(cols) > 1:
        plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv('df1.csv')
    # plot_series(df, 'UR')
    # plot_series(df, 'FEDR')
    # plot_series(df, 'GDP')
    # plot_series(df, 'CPI')
    # plot_series(df, 'PCE')
    plot_series(df, "male_ur", "female_ur", "black_ur", "hisp_ur","ur_1624","ur_2024","ur_2554","ur_cg_2534")
    plot_series(df, "PCE","CPI", "GDP", "FEDR")
if __name__ == '__main__':
    main()