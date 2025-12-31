import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import statsmodels.api as sm


def show_info(df):

    print(f'\nInfo are as follows \n')
    print(df)
    print(df.columns)
    print(df.describe())
    print(df.info())
    print(df.shape)

def clean_macro(df):

    # drop the starting label columns
    df = df.drop(df.index[0]).reset_index(drop=True)
    df = df.drop(df.index[0]).reset_index(drop=True)


    # rename each column with natural language(replace tickers with natural language)
    df.columns = ['Date',
                  'Financial Stress Index 4.0',
                  'HY Corporate Bond Yield Spread',
                  'Hourly Earnings %Change MoM',
                  'Average Hourly Earnings 1982-1984 USD YoY SA',
                  'Job Openings By Industry Total',
                  'Labor Force Participation Rate',
                  'PCE YOY$',
                  'CPI Urban Consumer',
                  'CPI Urban Consumer YOY',
                  'ISM Manufacturing PMI',
                  'US Industrial Production MOM',
                  'GDP',
                  'U-3 US Unemployment Rate Total',
                  'Effective Fed Funds Rate']

    # drop the useless columns, keep regressors
    df = df.drop('Financial Stress Index 4.0', axis = 1)
    df = df.drop('HY Corporate Bond Yield Spread',axis = 1)
    df = df.drop('Hourly Earnings %Change MoM',axis = 1)
    df = df.drop('Average Hourly Earnings 1982-1984 USD YoY SA',axis = 1)
    df = df.drop('Job Openings By Industry Total',axis = 1)
    df = df.drop('Labor Force Participation Rate',axis = 1)
    df = df.drop('CPI Urban Consumer',axis = 1)
    df = df.drop('ISM Manufacturing PMI',axis = 1)
    df = df.drop('US Industrial Production MOM',axis = 1)
    df = df.drop('Effective Fed Funds Rate',axis = 1)


    # convert datatype to float
    df = df.astype({col: "float" for col in df.columns if col != "Date"})
    # keep only monthly data
    df = df.dropna(subset=["U-3 US Unemployment Rate Total"]).reset_index(drop=True)

    # imputation QoQ GDP to MOM GDP
    df["GDP"] = df["GDP"].astype(float)
    df["GDP"] = df["GDP"].bfill().ffill()


    return df

def main():
    # read the data
    # Macro dataset
    macro = pd.read_csv('MacroData.csv')
    # read in the 2 yr yield & 10 yr yield
    DGS2 = pd.read_csv('DGS2.csv')
    DGS10 = pd.read_csv('DGS10.csv')
    # UR by categories
    male_ur = pd.read_csv('LNS14000001.csv')
    female_ur = pd.read_csv('LNS14000002.csv')
    black_ur = pd.read_csv('UR - Black.csv')
    hisp_ur = pd.read_csv('UR - Hispanic.csv')
    # UR for 16-24
    ur_1624 = pd.read_csv('UR 16-24.csv')
    # UR for 20-24
    ur_2024 = pd.read_csv('UR 20-24.csv')
    # UR for 25-24
    ur_2524 = pd.read_csv('UR 25-54.csv')
    # UR for College Graduates 25-34
    ur_cg_2534 = pd.read_csv('UR for College Graduates 25-34.csv')

    # effective fed funds rate
    FED = pd.read_csv('FEDR.csv')


    # compile dfs in a list
    df_list = [macro,
               DGS2,
               DGS10,
               male_ur,
               female_ur,
               black_ur,
               hisp_ur,
               ur_1624,
               ur_2024,
               ur_2524,
               ur_cg_2534]

    # list for unemployment rate
    ur_list = [male_ur,
               female_ur,
               black_ur,
               hisp_ur,
               ur_1624,
               ur_2024,
               ur_2524,
               ur_cg_2534]

    # print each data's info
    for df in ur_list:
        show_info(df)


    # data cleaning


    # print(FED.head())
    FED = FED.drop(FED.index[0]).reset_index(drop=True)
    FED = FED.drop(FED.index[0]).reset_index(drop=True)
    FED = FED.drop(FED.index[0]).reset_index(drop=True)
    FED = FED.iloc[::-1]
    FED = FED.reset_index(drop=True)
    print(FED.head())
    print(FED.info())

    # clean macro
    macro = clean_macro(macro)
    macro = macro.merge(FED[["FEDFUNDS"]], how="left", left_index=True, right_index=True)
    macro.columns = ['Date',
                   'PCE',
                   'CPI',
                   'GDP',
                   'UR',
                   'FEDR']
    show_info(macro)

    # print each data's info
    # for df in df_list:
    #   show_info(df)
    # print(macro['GDP'])

    # create a new df for ur
    cols = [df.iloc[:, 1] for df in ur_list]
    ur_df = pd.concat(cols, axis=1)
    ur_df = ur_df.iloc[3:].iloc[::-1].reset_index(drop=True)
    ur_df.columns = ['male_ur',
               'female_ur',
               'black_ur',
               'hisp_ur',
               'ur_1624',
               'ur_2024',
               'ur_2554',
               'ur_cg_2534']

    show_info(ur_df)

    merged = macro.merge(ur_df, left_index=True, right_index=True, how="left")
    merged["Date"] = pd.to_datetime(merged["Date"])


    # define COVID start and end dates
    covid_start = pd.Timestamp('2020-02-29')
    covid_end = pd.Timestamp('2023-03-31')

    # create dummy variable (1 during COVID period, otherwise 0)
    merged['covid_dummy'] = ((merged['Date'] >= covid_start) & (merged['Date'] <= covid_end)).astype(int)

    show_info(merged)
    # export to a separate .csv file
    merged.to_csv("df1.csv", index=False)








if __name__ == '__main__':
    main()
