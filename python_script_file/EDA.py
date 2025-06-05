#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import geopandas as gpd
import os


# In[2]:


growth_rates_emissions_energy_prod_income_level_country_df = pd.read_csv("../datasets/growth_rates_emissions_energy_prod_income_level_country_df.csv", index_col=None)

# growth_rates_emissions_energy_prod_income_level_country_df = pd.read_csv("/content/growth_rates_emissions_energy_prod_income_level_country_df.csv", index_col=None)


# In[3]:


growth_rates_emissions_energy_prod_income_level_country_df.head()


# In[4]:


growth_rates_emissions_energy_prod_income_level_country_df.shape


# In[5]:


growth_rates_emissions_energy_prod_income_level_country_df.isnull().sum()


# In[6]:


series_names = growth_rates_emissions_energy_prod_income_level_country_df['Series Name'].unique()
print(series_names)


# In[7]:


summary_stats = growth_rates_emissions_energy_prod_income_level_country_df.describe().transpose()

summary_stats['missing_values'] = growth_rates_emissions_energy_prod_income_level_country_df.isnull().sum()
summary_stats['data_type'] = growth_rates_emissions_energy_prod_income_level_country_df.dtypes

summary_stats


# ## Income Level Visualisation

# In[9]:


plt.figure(figsize=(10, 6))
unique_countries = growth_rates_emissions_energy_prod_income_level_country_df[["Country Name", "Income Level"]].drop_duplicates()

sns.countplot(y="Income Level", data=unique_countries, hue="Income Level", palette="coolwarm", legend=False,
              order=unique_countries["Income Level"].value_counts().index)
plt.title("Number of Unique Countries per Income Level")
plt.xlabel("Count")
plt.ylabel("Income Level")
plt.show()


# In[10]:


plt.figure(figsize=(8, 8))

unique_countries["Income Level"].value_counts().plot.pie(autopct="%1.1f%%", colors=["red", "blue", "green"])

plt.title("Proportion of Unique Countries by Income Level")
plt.ylabel("")
plt.show()


# ## Top Series Visualisations

# In[12]:


# Number of unique series in the dataset
num_series = growth_rates_emissions_energy_prod_income_level_country_df["Series Name"].nunique()
print(f"Total Unique Series: {num_series}")

# Count series occurrences
series_counts = growth_rates_emissions_energy_prod_income_level_country_df["Series Name"].value_counts()

# Display the top series
print("Top 10 Most Frequent Series:\n", series_counts.head(10))


# In[13]:


plt.figure(figsize=(12, 6))

# Convert series_counts into a DataFrame
series_df = series_counts.head(15).reset_index()
series_df.columns = ["Series Name", "Count"]

sns.barplot(y="Series Name", x="Count", hue="Series Name", data=series_df, palette="coolwarm", legend=False)

plt.title("Top 15 Most Frequent Economic Indicators (Series Names)")
plt.xlabel("Count")
plt.ylabel("Series Name")
plt.show()


# ## Groupby income level and list the countries

# In[15]:


# Group by Income Level and list unique countries
income_groups = growth_rates_emissions_energy_prod_income_level_country_df.groupby("Income Level")["Country Name"].unique()

# Convert to dictionary for easy access
income_level_countries = {level: list(countries) for level, countries in income_groups.items()}

# Convert to DataFrame for better visualization
income_level_df = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in income_level_countries.items()]))
income_level_df.head()


# --------------------------------------------------------------------------------------------------------------------------------------

# ### Correlation between different series
# This will directly contribute to my research questions different parts, and correlate between renewable energy adoption, energy efficiency, sectoral economic growth, and variation of CO2 emissions levels.

# 1. Renewable Energy Adoption (from Energy Production & Use)
#   - Renewable electricity output (% of total electricity output)
# 
#   - Electricity production from renewable sources, excluding hydroelectric (%)
# 
#   - Renewable energy consumption (% of total final energy consumption)
# 
#   - Combustible renewables and waste (% of total energy)
# 
# 2. Energy Efficiency (from Energy Production & Use)
#   - Energy intensity level of primary energy (MJ/$2017 PPP GDP)
# 
#   - GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)
# 
#   - Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)
# 
#   - Fossil fuel energy consumption (% of total)
# 
# 3. Sectoral Economic Growth (from Growth Rates)
#   - Agriculture, forestry, and fishing, value added (annual % growth)
# 
#   - Industry (including construction), value added (annual % growth)
# 
#   - Manufacturing, value added (annual % growth)
# 
#   - Services, value added (annual % growth)
# 
# 4. CO₂ Emissions & Environmental Impact (from Emissions)
#   - Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)
# 
#   - CO₂ emissions per capita
# 
#   - CO₂ emissions from Power, Industry, Transport, and Agriculture
# 
#   - Total greenhouse gas emissions excluding LULUCF
# 
#   - Carbon intensity of GDP

# #### Renewable Energy Adoption

# In[20]:


renewable_series = [
    "Renewable electricity output (% of total electricity output)",
    "Electricity production from renewable sources, excluding hydroelectric (% of total)",
    "Renewable energy consumption (% of total final energy consumption)",
    "Combustible renewables and waste (% of total energy)"
]


# #### Energy Efficiency

# In[22]:


efficiency_series = [
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)",
    "GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)",
    "Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)",
    "Fossil fuel energy consumption (% of total)"
]


# #### Sectoral Economic Growth

# In[24]:


growth_series = [
    "Agriculture, forestry, and fishing, value added (annual % growth)",
    "Industry (including construction), value added (annual % growth)",
    "Manufacturing, value added (annual % growth)",
    "Services, value added (annual % growth)"
]


# #### Emissions

# In[26]:


emission_series = [
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
    "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
    "Carbon intensity of GDP (kg CO2e per 2021 PPP $ of GDP)",
    "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
    "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"
]


# In[27]:


all_research_series = renewable_series + efficiency_series + growth_series + emission_series


# In[51]:


series_short_form_map = {
    # Renewable
    "Renewable electricity output (% of total electricity output)": "REO",
    "Electricity production from renewable sources, excluding hydroelectric (% of total)": "EPR",
    "Renewable energy consumption (% of total final energy consumption)": "REC",
    "Combustible renewables and waste (% of total energy)": "CRW",
    # Efficiency
    "Energy intensity level of primary energy (MJ/$2017 PPP GDP)": "EI",
    "GDP per unit of energy use (constant 2021 PPP $ per kg of oil equivalent)": "GDP/EU",
    "Energy use (kg of oil equivalent) per $1,000 GDP (constant 2021 PPP)": "EU/GDP",
    "Fossil fuel energy consumption (% of total)": "FFEC",
    # Growth
    "Agriculture, forestry, and fishing, value added (annual % growth)": "AgriGrowth",
    "Industry (including construction), value added (annual % growth)": "IndGrowth",
    "Manufacturing, value added (annual % growth)": "ManufGrowth",
    "Services, value added (annual % growth)": "ServGrowth",
    # Emissions
    "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)": "CO2T",
    "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)": "CO2pc",
    "Carbon intensity of GDP (kg CO2e per 2021 PPP $ of GDP)": "CIGDP",
    "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)": "GHGtotal",
    "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)": "GHGpc"
}


# ### Sector-Based Correlation (Pearson, Spearman, Kendall)

# In[55]:


year_columns = [col for col in growth_rates_emissions_energy_prod_income_level_country_df.columns if col.startswith("19") or col.startswith("20")]

income_levels = ["Low income", "Lower middle income", "Upper middle income"]

sector_series_dict = {
    "Agriculture": ["Agriculture, forestry, and fishing, value added (annual % growth)"],
    "Industry": [
        "Industry (including construction), value added (annual % growth)",
        "Manufacturing, value added (annual % growth)"
    ],
    "Services": ["Services, value added (annual % growth)"]
}

for income in income_levels:
    for sector, sector_growth_series in sector_series_dict.items():
        sector_related_series = renewable_series + efficiency_series + emission_series + sector_growth_series

        filtered_df = growth_rates_emissions_energy_prod_income_level_country_df[
            (growth_rates_emissions_energy_prod_income_level_country_df["Income Level"] == income) &
            (growth_rates_emissions_energy_prod_income_level_country_df["Series Name"].isin(sector_related_series))
        ].copy()

        filtered_df["Series Avg"] = filtered_df[year_columns].mean(axis=1)

        pivot_df = filtered_df.pivot_table(
            index="Country Name", columns="Series Name", values="Series Avg"
        ).dropna()

        if pivot_df.shape[0] < 3:
            print(f"Skipped {sector} - {income} (not enough data)")
            continue

        for method in ['pearson']: # for method in ['pearson', 'spearman', 'kendall']:
            corr_matrix = pivot_df.corr(method=method)
            # Rename rows and columns to short forms
            short_corr_matrix = corr_matrix.rename(index=series_short_form_map, columns=series_short_form_map)

            print(f"{sector} sector correlation matrix | {income} | {method}\n",
                  short_corr_matrix[[series_short_form_map.get(s, s) for s in sector_growth_series]])

            plt.figure(figsize=(12, 9))
            sns.heatmap(short_corr_matrix, cmap="coolwarm", annot=True, fmt=".2f")
            plt.title(f"{sector} Sector Correlation | {income} | {method.title()} Method")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Sava plot
            filename = f"../plots/{sector}_{income}_{method}_correlation_heatmap.png".replace(" ", "_")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()


# In[ ]:




# Commit 14 - Visualize temporal evolution of REC effects across decades
