import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


def model_one(df):
    # regression Model No.1
    X = df[["PCE", "CPI", "GDP", "FEDR"]]
    X = sm.add_constant(X)
    y = df["UR"]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    #  Corr Matrix
    corr_matrix = X.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    # Corr Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Model 1")
    plt.show()

    # VIF table
    X_with_const = sm.add_constant(X)

    vif = pd.DataFrame()
    vif["variable"] = X_with_const.columns
    vif["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                  for i in range(X_with_const.shape[1])]

    print("\nVIF Table:\n")
    print(vif)

    # durbin-watson test
    dw = durbin_watson(model.resid)
    print(dw)

    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Squared residuals
    residuals_sq = residuals ** 2

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_vals, residuals_sq, alpha=0.7)
    plt.xlabel("Predicted UR (Ŷ)")
    plt.ylabel("Squared Residuals (ε̂²)")
    plt.title("Residuals Squared vs. Fitted Values")
    plt.grid(True)
    plt.show()

    return model

def model_two(df):
    # regression Model No.1
    X = df[["PCE","CPI", "GDP", "FEDR","male_ur","female_ur","black_ur","hisp_ur","ur_1624","ur_2024","ur_2554","ur_cg_2534"]]
    X = sm.add_constant(X)
    y = df["UR"]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    #  Corr Matrix
    corr_matrix = X.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    # Corr Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Model 2")
    plt.show()

    # VIF table
    X_with_const = sm.add_constant(X)

    vif = pd.DataFrame()
    vif["variable"] = X_with_const.columns
    vif["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                  for i in range(X_with_const.shape[1])]

    print("\nVIF Table:\n")
    print(vif)

    # durbin-watson test
    dw = durbin_watson(model.resid)
    print(dw)

    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Squared residuals
    residuals_sq = residuals ** 2

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_vals, residuals_sq, alpha=0.7)
    plt.xlabel("Predicted UR (Ŷ)")
    plt.ylabel("Squared Residuals (ε̂²)")
    plt.title("Residuals Squared vs. Fitted Values")
    plt.grid(True)
    plt.show()

    return model

def model_three(df):
    # regression Model No.1
    X = df[["PCE", "CPI", "GDP", "FEDR","covid_dummy"]]
    X = sm.add_constant(X)
    y = df["UR"]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    #  Corr Matrix
    corr_matrix = X.corr()
    print("Correlation Matrix:")
    print(corr_matrix)
    # Corr Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix for Model 3")
    plt.show()

    # VIF table
    X_with_const = sm.add_constant(X)

    vif = pd.DataFrame()
    vif["variable"] = X_with_const.columns
    vif["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                  for i in range(X_with_const.shape[1])]

    print("\nVIF Table:\n")
    print(vif)

    # durbin-watson test
    dw = durbin_watson(model.resid)
    print(dw)

    # Get fitted values and residuals
    fitted_vals = model.fittedvalues
    residuals = model.resid

    # Squared residuals
    residuals_sq = residuals ** 2

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(fitted_vals, residuals_sq, alpha=0.7)
    plt.xlabel("Predicted UR (Ŷ)")
    plt.ylabel("Squared Residuals (ε̂²)")
    plt.title("Residuals Squared vs. Fitted Values")
    plt.grid(True)
    plt.show()

    return model


def run_hypothesis_tests(results):
    """
    Interactive hypothesis testing module.
    User selects test type:
      1 = Single β t-test
      2 = Overall F-test
      3 = Subset F-test
      q = Quit (only exit this function)
    """

    params = list(results.params.index)
    print("\nAvailable coefficients:", params)

    # -----------------------------
    # 1. Get alpha (ONE TIME ONLY)
    # -----------------------------
    while True:
        user_alpha = input("\nEnter significance level alpha (e.g., 0.05): ").strip()
        try:
            alpha = float(user_alpha)
            if 0 < alpha < 1:
                break
            else:
                print("Alpha must be between 0 and 1. Try again.")
        except:
            print("Invalid number. Try again.")

    # -----------------------------
    # MAIN INTERACTIVE LOOP
    # -----------------------------
    while True:
        print("\nChoose a hypothesis test:")
        print("  1 = Single β t-test")
        print("  2 = Overall F-test")
        print("  3 = Subset F-test")
        print("  q = Quit to main program")

        choice = input("Enter choice: ").strip().lower()

        # -------------------------
        # QUIT THIS FUNCTION ONLY
        # -------------------------
        if choice == 'q':
            print("\nExiting hypothesis-testing module.\n")
            return  # DOES NOT terminate main program

        # ====================================================
        # 1. SINGLE β t-test
        # ====================================================
        elif choice == '1':
            while True:
                beta = input("\nEnter coefficient for t-test (or Enter to cancel): ").strip()

                if beta == "":
                    print("Canceled single β test.")
                    break

                if beta not in params:
                    print(f"'{beta}' not found. Available: {params}")
                    continue  # ask again

                # Perform test
                t_stat = results.tvalues[beta]
                p_val = results.pvalues[beta]

                print(f"\nSingle β t-test for H0: {beta} = 0")
                print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4g}")

                if p_val < alpha:
                    print(f"p < {alpha} → **Reject H0** → {beta} IS significant.")
                else:
                    print(f"p ≥ {alpha} → **Fail to reject H0** → {beta} is NOT significant.")

                break  # exit single-test loop

        # ====================================================
        # 2. OVERALL F-TEST
        # ====================================================
        elif choice == '2':
            f_stat = results.fvalue
            f_pval = results.f_pvalue

            print("\nOverall Model F-test")
            print("H0: All slope coefficients = 0")
            print(f"F-statistic = {f_stat:.4f}, p-value = {f_pval:.4g}")

            if f_pval < alpha:
                print(f"p < {alpha} → **Reject H0** → Model IS overall significant.")
            else:
                print(f"p ≥ {alpha} → **Fail to reject H0** → Model is NOT overall significant.")

        # ====================================================
        # 3. SUBSET F-TEST
        # ====================================================
        elif choice == '3':
            while True:
                subset_input = input(
                    "\nEnter coefficients for subset F-test (comma-separated), or Enter to cancel: "
                ).strip()

                if subset_input == "":
                    print("Canceled subset test.")
                    break

                subset_vars = [v.strip() for v in subset_input.split(",") if v.strip()]

                invalid = [v for v in subset_vars if v not in params]
                if invalid:
                    print(f"Invalid: {invalid}. Valid: {params}")
                    continue  # ask again

                hypothesis = " , ".join([f"{v} = 0" for v in subset_vars])

                try:
                    ftest_res = results.f_test(hypothesis)
                    f_stat = float(ftest_res.fvalue)
                    p_val = float(ftest_res.pvalue)
                except Exception as e:
                    print("Error running F-test:", e)
                    continue

                print(f"\nSubset F-test for H0: {hypothesis}")
                print(f"F-statistic = {f_stat:.4f}, p-value = {p_val:.4g}")

                if p_val < alpha:
                    print(f"p < {alpha} → **Reject H0** → {subset_vars} ARE jointly significant.")
                else:
                    print(f"p ≥ {alpha} → **Fail to reject H0** → {subset_vars} are NOT jointly significant.")

                break  # exit subset-test loop

        # ====================================================
        # INVALID INPUT
        # ====================================================
        else:
            print("Invalid choice. Please enter 1, 2, 3, or q.")
            continue

def model_menu(models):
    """
    Menu to choose which model's results to use,
    then call run_hypothesis_tests(results) on it.

    `models` is a dict like:
        {"1": ("m1", m1_results),
         "2": ("m2", m2_results),
         "3": ("m3", m3_results)}
    """
    while True:
        print("\n=== MODEL SELECTION MENU ===")
        for key, (label, _) in models.items():
            print(f"  {key} = {label}")
        print("  q = Quit model menu")

        choice = input("Choose a model: ").strip().lower()

        if choice == "q":
            print("\nExiting model menu.\n")
            break

        if choice in models:
            label, results = models[choice]
            print(f"\nYou selected {label}.")
            run_hypothesis_tests(results)   # <- call your test menu here
        else:
            print("Invalid choice. Please try again.")

def main():
    df1 = pd.read_csv('df1.csv')

    print(df1)
    print(df1.info())

    m1 = model_one(df1)
    print(f"\n\n\n\n\n\n\n")
    m2 = model_two(df1)
    m3 = model_three(df1)

    # Hypothesis testing
    models = {
        "1": ("Model 1 (m1)", m1),
        "2": ("Model 2 (m2)", m2),
        "3": ("Model 3 (m3)", m3),
    }

    model_menu(models)



if __name__ == '__main__':
    main()