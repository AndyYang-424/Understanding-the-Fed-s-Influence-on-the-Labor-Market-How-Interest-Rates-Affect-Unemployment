import pandas as pd

# -----------------------------
# 1. Load dataset
# -----------------------------
file_path = "df1.csv"
df = pd.read_csv(file_path)

# -----------------------------
# 2. Remove the Date column
# -----------------------------
df_numeric = df.drop(columns=["Date"], errors="ignore")

# Keep only numeric columns (important for stats)
df_numeric = df_numeric.select_dtypes(include="number")

# -----------------------------
# 3. Compute summary statistics
# -----------------------------
summary = pd.DataFrame({
    "Observations": df_numeric.count(),
    "Mean": df_numeric.mean(),
    "Std. Dev.": df_numeric.std(),
    "Min": df_numeric.min(),
    "Max": df_numeric.max()
})

# -----------------------------
# 4. Display final table
# -----------------------------
print("\n=== SUMMARY STATISTICS (excluding Date) ===\n")
print(summary)

# Optional: Save to Excel/CSV
summary.to_excel("summary_stats.xlsx")
