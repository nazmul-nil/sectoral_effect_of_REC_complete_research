#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# Load the GMM base dataset we just created
data_path = "../datasets/growth_rates_energy_gmm_base.csv"
df = pd.read_csv(data_path)

print("GMM Base Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Countries: {df['Country Name'].nunique()}")
print(f"Time periods: {df['year'].min()} - {df['year'].max()}")

# Step 1: Prepare panel structure
print("\n" + "="*60)
print("STEP 1: Preparing panel structure for lag generation")
print("="*60)

# Ensure proper sorting by country and year
df = df.sort_values(['Country Name', 'year'])

# Set panel index for efficient groupby operations
df.set_index(['Country Name', 'year'], inplace=True)

print("✅ Panel structure prepared")

# Step 2: Define variables for GMM lagging
print("\n" + "="*60)
print("STEP 2: Defining variables for GMM lag generation")
print("="*60)

# Key variables for GMM estimation
# These are the endogenous and predetermined variables that need lags as instruments
lag_vars = [
    'GDPgrowth',      # Dependent variable (economic growth)
    'AgriGrowth',     # Sectoral growth variables
    'IndGrowth', 
    'ServGrowth',
    'REC',            # Renewable energy consumption
    'EI',             # Energy intensity  
    'AccessElec',     # Access to electricity
    'PM2.5'           # Air pollution measure
]

# Verify all variables exist
available_vars = [var for var in lag_vars if var in df.columns]
missing_vars = [var for var in lag_vars if var not in df.columns]

print(f"Variables to lag: {len(available_vars)}")
for var in available_vars:
    print(f"✅ {var}")

if missing_vars:
    print(f"\nMissing variables (will skip): {missing_vars}")

# Step 3: Generate 1-3 period lags
print("\n" + "="*60)
print("STEP 3: Generating lagged variables (1-3 periods)")
print("="*60)

lag_count = 0
for var in available_vars:
    print(f"\nCreating lags for {var}:")
    for lag in range(1, 4):  # Create 1, 2, and 3 period lags
        lag_col_name = f'{var}_lag{lag}'
        df[lag_col_name] = df.groupby(level=0)[var].shift(lag)
        
        # Count non-missing values
        non_missing = df[lag_col_name].count()
        print(f"  • {lag_col_name}: {non_missing:,} observations")
        lag_count += 1

print(f"\n✅ Created {lag_count} lagged variables")

# Step 4: Check panel balance and missing data
print("\n" + "="*60)
print("STEP 4: Analyzing panel balance and data availability")
print("="*60)

# Show observations per country before and after lagging
country_obs = df.groupby(level=0).size()
print(f"Observations per country - Min: {country_obs.min()}, Max: {country_obs.max()}, Mean: {country_obs.mean():.1f}")

# Check how many countries have sufficient data for GMM (need at least lag1)
lag1_vars = [f'{var}_lag1' for var in available_vars]
countries_with_lags = df.groupby(level=0)[lag1_vars].count().min(axis=1)
sufficient_countries = (countries_with_lags >= 1).sum()
print(f"Countries with at least 1 lag observation: {sufficient_countries}/{len(country_obs)}")

# Step 5: Create clean GMM dataset
print("\n" + "="*60)
print("STEP 5: Creating clean GMM-ready dataset")
print("="*60)

# Option 1: Keep all observations (recommended for unbalanced panel GMM)
df_gmm_all = df.copy()

# Option 2: Keep only observations with at least lag1 for key variables
key_lag1_vars = [f'{var}_lag1' for var in ['GDPgrowth', 'AgriGrowth', 'IndGrowth', 'ServGrowth']]
available_key_lag1 = [var for var in key_lag1_vars if var in df.columns]

df_gmm_clean = df.dropna(subset=available_key_lag1, how='all')

print(f"Dataset options:")
print(f"• All observations (unbalanced): {len(df_gmm_all):,} obs")
print(f"• Clean observations (with key lags): {len(df_gmm_clean):,} obs") 
print(f"• Dropped due to missing key lags: {len(df_gmm_all) - len(df_gmm_clean):,} obs")

# Use the clean dataset for GMM
df_gmm_final = df_gmm_clean.copy()

# Step 6: Add first differences for difference GMM
print("\n" + "="*60)
print("STEP 6: Adding first differences for difference GMM")
print("="*60)

# Create first differences for key variables
diff_vars = available_vars.copy()
for var in diff_vars:
    diff_col_name = f'D_{var}'
    df_gmm_final[diff_col_name] = df_gmm_final.groupby(level=0)[var].diff()
    non_missing = df_gmm_final[diff_col_name].count()
    print(f"• {diff_col_name}: {non_missing:,} observations")

print("✅ First differences created for difference GMM")

# Step 7: Final dataset preparation
print("\n" + "="*60)
print("STEP 7: Final dataset preparation and summary")
print("="*60)

# Reset index for saving
df_gmm_final.reset_index(inplace=True)

# Create summary statistics
print(f"📊 FINAL GMM-READY DATASET SUMMARY:")
print(f"• Total observations: {len(df_gmm_final):,}")
print(f"• Countries: {df_gmm_final['Country Name'].nunique()}")
print(f"• Time periods: {df_gmm_final['year'].min()} - {df_gmm_final['year'].max()}")
print(f"• Average observations per country: {len(df_gmm_final)/df_gmm_final['Country Name'].nunique():.1f}")

# Count different types of variables
level_vars = available_vars
lag_vars_created = [col for col in df_gmm_final.columns if '_lag' in col]
diff_vars_created = [col for col in df_gmm_final.columns if col.startswith('D_')]

print(f"\n🏷️  Variable Types:")
print(f"• Level variables: {len(level_vars)}")
print(f"• Lagged variables: {len(lag_vars_created)}")
print(f"• First differences: {len(diff_vars_created)}")
print(f"• Total variables: {len(df_gmm_final.columns)}")

# Step 8: Save GMM-ready dataset
print("\n" + "="*60)
print("STEP 8: Saving GMM-ready dataset")
print("="*60)

output_path = "../datasets/growth_rates_energy_gmm_ready.csv"
df_gmm_final.to_csv(output_path, index=False)
print(f"✅ GMM-ready dataset saved at: {output_path}")

# Step 9: Display variable lists for GMM specification
print("\n" + "="*60)
print("STEP 9: GMM estimation variable reference")
print("="*60)

print("📋 VARIABLE REFERENCE FOR GMM ESTIMATION:")

print(f"\n🎯 LEVEL VARIABLES (for System GMM):")
for i, var in enumerate(level_vars, 1):
    print(f"{i:2d}. {var}")

print(f"\n⏰ LAG VARIABLES (for instruments):")
lag_by_variable = {}
for var in available_vars:
    lags = [f'{var}_lag{i}' for i in range(1, 4) if f'{var}_lag{i}' in df_gmm_final.columns]
    if lags:
        lag_by_variable[var] = lags

for var, lags in lag_by_variable.items():
    print(f"• {var}: {', '.join(lags)}")

print(f"\n📈 FIRST DIFFERENCES (for Difference GMM):")
for i, var in enumerate(diff_vars_created, 1):
    print(f"{i:2d}. {var}")

print(f"\n💡 CONTROL VARIABLES:")
print("• Income Level (categorical)")
print("• year (time effects)")

# Step 10: Sample preview
print("\n" + "="*60)
print("STEP 10: Dataset preview")
print("="*60)

# Show sample with key variables
sample_cols = ['Country Name', 'year', 'Income Level'] + level_vars[:3] + \
              [f'{available_vars[0]}_lag1', f'{available_vars[0]}_lag2'] + \
              [f'D_{available_vars[0]}']

available_sample_cols = [col for col in sample_cols if col in df_gmm_final.columns]
print(f"Sample of GMM-ready dataset (showing {len(available_sample_cols)} columns):")
print(df_gmm_final[available_sample_cols].head(10))

print(f"\n🎉 SUCCESS! Dataset is ready for GMM estimation.")
print(f"📄 Use this dataset for:")
print(f"   • System GMM (Arellano-Bond-Bover/Blundell-Bond)")
print(f"   • Difference GMM (Arellano-Bond)")
print(f"   • Two-step estimation with robust standard errors")

# Data quality check
print(f"\n📊 DATA QUALITY CHECK:")
for var in level_vars:
    if var in df_gmm_final.columns:
        missing_pct = (df_gmm_final[var].isna().sum() / len(df_gmm_final)) * 100
        print(f"• {var}: {missing_pct:.1f}% missing")

print(f"\n✨ Ready for heterogeneous renewable energy adoption analysis!")

