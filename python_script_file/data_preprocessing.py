#!/usr/bin/env python
# coding: utf-8

# In[608]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[609]:


df = pd.read_csv('../datasets/P_Data_Extract_From_World_Development_Indicators/0bcc63a9-526c-4673-98ef-4f8ebcf0fe7c_Data.csv')


# In[610]:


df.head()


# In[611]:


df.info()


# In[612]:


df.shape


# In[613]:


df.isnull().sum()


# In[614]:


df.columns


# In[615]:


series_name = df['Series Name'].value_counts()

print(series_name)


# In[616]:


df['Country Name'].value_counts()


# In[617]:


df.replace("..", pd.NA, inplace=True)


# In[618]:


df.head()


# In[619]:


df.isnull().sum()


# In[620]:


# Selecting only year columns
year_columns = df.columns[4:]
df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')


# In[621]:


df.info()


# In[622]:


df[year_columns] = df.groupby(["Country Name", "Series Name"])[year_columns].transform(lambda x: x.bfill().ffill())


# In[623]:


df["2023 [YR2023]"] = df["2023 [YR2023]"].where(df["2023 [YR2023]"] < df["2022 [YR2022]"] * 1.5, df["2022 [YR2022]"])


# In[624]:


df.isnull().sum()


# In[625]:


# df[year_columns] = df.groupby(["Country Name", "Series Name"])[year_columns].transform(lambda x: x.fillna(x.mean()))

df[year_columns] = df[year_columns].apply(lambda row: row.fillna(row.mean()), axis=1)


# In[626]:


missing_threshold = 0.5 * len(year_columns)
df = df.dropna(thresh=missing_threshold, axis=0)


# In[627]:


df.shape


# In[628]:


df.isnull().sum()


# In[629]:


df.head()


# In[630]:


df.columns = [col.replace(" [YR", "_").replace("]", "") if "YR" in col else col for col in df.columns]


# In[631]:


df.reset_index(drop=True, inplace=True)


# In[632]:


df.head()


# In[633]:


df.shape


# In[634]:


df.to_csv('../datasets/cleaned_dataset.csv')


# In[635]:


# World Bank, 2025
# World Bank country classifications by income level for 2024-2025
# https://databank.worldbank.org/data/download/site-content/CLASS.xlsx

url = "https://databank.worldbank.org/data/download/site-content/CLASS.xlsx"

column_names = ["Country Name", "Country Code", "Region", "Income Level", "Lending Category", "Extra"]

income_df = pd.read_excel(url, sheet_name="List of economies", skiprows=3, header=None,  names=column_names)


# In[636]:


income_df.head(100)


# In[637]:


income_df = income_df[["Country Name", "Country Code", "Income Level"]]


# In[638]:


growth_rates_emissions_energy_prod_income_level_country_df = df.merge(income_df, on=["Country Name", "Country Code"], how="left")


# In[639]:


growth_rates_emissions_energy_prod_income_level_country_df.head()


# In[640]:


print(growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].value_counts())


# In[641]:


missing_countries = growth_rates_emissions_energy_prod_income_level_country_df[growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].isna()]["Country Name"].unique()
print("Countries with missing income classification:", missing_countries)


# In[642]:


regions_to_remove = [
    "World", "Sub-Saharan Africa", "OECD members", "East Asia & Pacific", "Europe & Central Asia",
    "Middle East & North Africa", "Latin America & Caribbean", "South Asia", "North America",
    "Fragile and conflict affected situations", "Least developed countries: UN classification",
    "Low & middle income", "Lower middle income", "Middle income", "Upper middle income",
    "Heavily indebted poor countries (HIPC)", "Small states", "IBRD only", "IDA & IBRD total",
    "IDA blend", "IDA only", "IDA total", "Euro area", "European Union",
    "East Asia & Pacific (IDA & IBRD countries)", "Europe & Central Asia (IDA & IBRD countries)",
    "Latin America & the Caribbean (IDA & IBRD countries)",
    "Middle East & North Africa (IDA & IBRD countries)", "Sub-Saharan Africa (IDA & IBRD countries)",
    "Africa Eastern and Southern", "Africa Western and Central", "Arab World", "Caribbean small states",
    "Central Europe and the Baltics", "Early-demographic dividend", "East Asia & Pacific (excluding high income)",
    "Europe & Central Asia (excluding high income)", "High income", "Late-demographic dividend",
    "Latin America & Caribbean (excluding high income)", "Low income", "Middle East & North Africa (excluding high income)",
    "Other small states", "Pacific island small states", "Post-demographic dividend",
    "Pre-demographic dividend", "South Asia (IDA & IBRD)", "Sub-Saharan Africa (excluding high income)"
]

growth_rates_emissions_energy_prod_income_level_country_df = growth_rates_emissions_energy_prod_income_level_country_df[
    ~growth_rates_emissions_energy_prod_income_level_country_df["Country Name"].isin(regions_to_remove)
]

print("Remaining unique countries:", growth_rates_emissions_energy_prod_income_level_country_df["Country Name"].nunique())


# In[643]:


missing_countries = growth_rates_emissions_energy_prod_income_level_country_df[
    growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].isna()
]["Country Name"].unique()

print("Low/Middle-Income Countries with Missing Income Classification:", missing_countries)


# In[644]:


manual_income_levels = {
    "Cote d'Ivoire": "Lower middle income",
    "Sao Tome and Principe": "Lower middle income",
    "Turkiye": "Upper middle income",
    "Viet Nam": "Lower middle income",
    "Afghanistan": "Low income",
    "Venezuela, RB": "Not classified",
    "Aruba": "High income",
    "Curacao": "High income",
    "Czechia": "Not classified"   
}

growth_rates_emissions_energy_prod_income_level_country_df["Income Level"] = growth_rates_emissions_energy_prod_income_level_country_df[
    "Country Name"
].map(manual_income_levels).fillna(growth_rates_emissions_energy_prod_income_level_country_df["Income Level"])

missing_countries_final = growth_rates_emissions_energy_prod_income_level_country_df[
    growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].isna()
]["Country Name"].unique()

print("Final missing countries:", missing_countries_final)


# In[645]:


# Remove High income countries

growth_rates_emissions_energy_prod_income_level_country_df = growth_rates_emissions_energy_prod_income_level_country_df[
    ~growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].isin(["High income", "Not classified"])
]


# In[646]:


growth_rates_emissions_energy_prod_income_level_country_df.tail()


# In[647]:


growth_rates_emissions_energy_prod_income_level_country_df['Income Level'].value_counts()


# In[648]:


growth_rates_emissions_energy_prod_income_level_country_df.shape


# In[649]:


growth_rates_emissions_energy_prod_income_level_country_df.info()


# In[650]:


growth_rates_emissions_energy_prod_income_level_country_df.isnull().sum()


# In[651]:


growth_rates_emissions_energy_prod_income_level_country_df['Income Level'].value_counts()


# In[652]:


unique_country_count = growth_rates_emissions_energy_prod_income_level_country_df["Country Name"].nunique()
print(f"Final unique country count: {unique_country_count}")


# In[653]:


duplicates = growth_rates_emissions_energy_prod_income_level_country_df.duplicated()
assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate rows!"
print("No duplicate rows found.")


# In[654]:


valid_income_levels = ["Low income", "Lower middle income", "Upper middle income"]
unique_income_levels = growth_rates_emissions_energy_prod_income_level_country_df["Income Level"].unique()

assert all(income in valid_income_levels for income in unique_income_levels), "There are invalid income levels in the dataset!"
print("Only low, lower middle, and upper middle income countries remain.")


# In[655]:


country_check = growth_rates_emissions_energy_prod_income_level_country_df[growth_rates_emissions_energy_prod_income_level_country_df[
    "Country Name"
].isin(["Bulgaria", "Palau", "Russian Federation"])]

country_check.tail()


# In[656]:


# This year, three countries—Bulgaria, Palau, and Russia—moved from the upper-middle-income to the high-income category (World Bank, 2025)
# World Bank country classifications by income level for 2024-2025
# https://blogs.worldbank.org/en/opendata/world-bank-country-classifications-by-income-level-for-2024-2025
# Remove Bulgaria, Palau, Russian Federation

growth_rates_emissions_energy_prod_income_level_country_df = growth_rates_emissions_energy_prod_income_level_country_df[
    ~growth_rates_emissions_energy_prod_income_level_country_df["Country Name"].isin(["Bulgaria", "Palau", "Russian Federation"])
]


# In[657]:


country_check = growth_rates_emissions_energy_prod_income_level_country_df[growth_rates_emissions_energy_prod_income_level_country_df[
    "Country Name"
].isin(["Bulgaria", "Palau", "Russian Federation"])]

country_check.head()


# In[658]:


growth_rates_emissions_energy_prod_income_level_country_df.shape


# In[659]:


growth_rates_emissions_energy_prod_income_level_country_df.isnull().sum()


# In[660]:


growth_rates_emissions_energy_prod_income_level_country_df.info()


# In[661]:


growth_rates_emissions_energy_prod_income_level_country_df.to_csv('../datasets/growth_rates_emissions_energy_prod_income_level_country_df.csv')

print("Merged Dataset with only Low, Lower middle, Upper income level countries saved successfully")

