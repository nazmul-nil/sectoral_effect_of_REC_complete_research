#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Load enhanced dataset
df = pd.read_csv("../datasets/enhanced_growth_rates_emissions_energy_prod_income_level_country_df.csv")

# Focus on key GMM-related indicators
key_series = [
    'Agriculture, forestry, and fishing, value added (annual % growth)',
    'Industry (including construction), value added (annual % growth)',
    'Services, value added (annual % growth)',
    'Renewable energy consumption (% of total final energy consumption)',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)',
    'Access to electricity (% of population)',
    'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)',
    'GDP growth (annual %)'
]

df_gmm_base = df[df["Series Name"].isin(key_series)].copy()

# Pivot dataset: rows = (country, year), columns = indicator values
df_panel = df_gmm_base.melt(
    id_vars=["Country Name", "Country Code", "Income Level", "Series Name"],
    var_name="year",
    value_name="value"
)

# Convert year to numeric
df_panel["year"] = df_panel["year"].astype(str).str.extract(r'(\d{4})').astype(int)

# Pivot to wide format: 1 row per (Country, year), columns = indicators
df_panel = df_panel.pivot_table(
    index=["Country Name", "Country Code", "Income Level", "year"],
    columns="Series Name",
    values="value"
).reset_index()

# Rename key columns for simplicity
rename_map = {
    'Agriculture, forestry, and fishing, value added (annual % growth)': 'AgriGrowth',
    'Industry (including construction), value added (annual % growth)': 'IndGrowth',
    'Services, value added (annual % growth)': 'ServGrowth',
    'Renewable energy consumption (% of total final energy consumption)': 'REC',
    'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': 'EI',
    'Access to electricity (% of population)': 'AccessElec',
    'PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)': 'PM2.5',
    'GDP growth (annual %)': 'GDPgrowth'
}

df_panel.rename(columns=rename_map, inplace=True)

# Save cleaned GMM base panel
df_panel.to_csv("../datasets/growth_rates_energy_gmm_base.csv", index=False)
print("✅ Saved cleaned GMM base panel to: growth_rates_energy_gmm_base.csv")

