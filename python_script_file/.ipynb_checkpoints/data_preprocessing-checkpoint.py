#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🔧 Enhanced Preprocessing for Panel Data Econometrics")
print("="*60)

# ============================================================================
# STEP 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("\n📊 Step 1: Loading and exploring raw data...")

df = pd.read_csv('../datasets/P_Data_Extract_From_World_Development_Indicators/0bcc63a9-526c-4673-98ef-4f8ebcf0fe7c_Data.csv')

print(f"Initial dataset shape: {df.shape}")
print(f"Number of series: {df['Series Name'].nunique()}")
print(f"Number of countries: {df['Country Name'].nunique()}")

# ============================================================================
# STEP 2: INITIAL CLEANING AND TYPE CONVERSION
# ============================================================================

print("\n🧹 Step 2: Initial cleaning and type conversion...")

# Convert World Bank missing value indicators to pandas NA
df.replace(["..", "...", ""], pd.NA, inplace=True)

# Identify year columns
year_columns = [col for col in df.columns if 'YR' in col and '[' in col]
non_year_columns = [col for col in df.columns if col not in year_columns]

print(f"Year columns identified: {len(year_columns)} columns from {year_columns[0]} to {year_columns[-1]}")

# Convert year columns to numeric
df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')

# Clean column names
df.columns = [col.replace(" [YR", "_").replace("]", "") if "YR" in col else col for col in df.columns]
year_columns = [col.replace(" [YR", "_").replace("]", "") for col in year_columns]

print("✅ Column names cleaned and data types converted")

# ============================================================================
# STEP 3: ENHANCED MISSING VALUE TREATMENT
# ============================================================================

print("\n🔄 Step 3: Enhanced missing value treatment...")

# Store original missing count for comparison
original_missing = df[year_columns].isnull().sum().sum()
print(f"Original missing values: {original_missing:,}")

# Method 1: Within-group temporal imputation (STANDARD PRACTICE)
print("Applying within-group temporal imputation...")
df[year_columns] = df.groupby(["Country Name", "Series Name"])[year_columns].transform(
    lambda x: x.bfill().ffill()
)

# Count missing after temporal imputation
after_temporal = df[year_columns].isnull().sum().sum()
print(f"Missing after temporal imputation: {after_temporal:,}")

# Method 2: REMOVED - Cross-series imputation (problematic for econometrics)
# This step from original code is removed as it can introduce spurious correlations

print("✅ Using only within-group temporal imputation (econometric best practice)")

# ============================================================================
# STEP 4: ENHANCED OUTLIER DETECTION AND TREATMENT
# ============================================================================

print("\n🎯 Step 4: Enhanced outlier detection and treatment...")

def detect_outliers_iqr(series, factor=1.5):
    """Detect outliers using IQR method"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (series < lower_bound) | (series > upper_bound)

def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold

# Apply systematic outlier detection for growth variables
growth_indicators = [col for col in df['Series Name'].unique() 
                    if pd.notna(col) and ('growth' in str(col).lower() or 'Growth' in str(col))]
outlier_summary = {}

print("Detecting outliers in growth indicators...")
for indicator in growth_indicators:
    indicator_data = df[df['Series Name'] == indicator]
    
    for year_col in year_columns:
        if year_col in indicator_data.columns:
            series = indicator_data[year_col].dropna()
            
            if len(series) > 10:  # Need sufficient data
                # Use IQR method for growth rates (more conservative)
                outliers_iqr = detect_outliers_iqr(series, factor=2.0)  # More conservative
                outlier_count = outliers_iqr.sum()
                
                if outlier_count > 0:
                    outlier_summary[f"{indicator}_{year_col}"] = outlier_count
                    
                    # Cap extreme outliers instead of removing them
                    upper_cap = series.quantile(0.95)
                    lower_cap = series.quantile(0.05)
                    
                    # Apply winsorization using safer indexing
                    mask = df['Series Name'] == indicator
                    mask_indices = df.index[mask]
                    
                    # Apply clipping to the specific indices
                    df.loc[mask_indices, year_col] = df.loc[mask_indices, year_col].clip(lower=lower_cap, upper=upper_cap)

total_outliers = sum(outlier_summary.values())
print(f"Total outliers detected and winsorized: {total_outliers}")
print("✅ Outlier treatment completed using statistical methods")

# ============================================================================
# STEP 5: DATA QUALITY VALIDATION
# ============================================================================

print("\n✅ Step 5: Data quality validation...")

# Check for impossible values in specific indicators
validation_rules = {
    'GDP growth': (-50, 50),  # GDP growth should be within reasonable bounds
    'population growth': (-10, 10),  # Population growth bounds
    'inflation': (-50, 1000),  # Inflation bounds
}

validation_issues = 0
for indicator_pattern, (min_val, max_val) in validation_rules.items():
    matching_series = [series for series in df['Series Name'].unique() 
                      if pd.notna(series) and indicator_pattern.lower() in str(series).lower()]
    
    for series_name in matching_series:
        mask = df['Series Name'] == series_name
        for year_col in year_columns:
            invalid_mask = (df.loc[mask, year_col] < min_val) | (df.loc[mask, year_col] > max_val)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                validation_issues += invalid_count
                # Set invalid values to NaN
                df.loc[mask & invalid_mask, year_col] = pd.NA

print(f"Data validation issues found and corrected: {validation_issues}")

# ============================================================================
# STEP 6: ENHANCED COUNTRY CLASSIFICATION
# ============================================================================

print("\n🌍 Step 6: Enhanced country classification...")

# Load World Bank income classifications
url = "https://databank.worldbank.org/data/download/site-content/CLASS.xlsx"
column_names = ["Country Name", "Country Code", "Region", "Income Level", "Lending Category", "Extra"]

try:
    income_df = pd.read_excel(url, sheet_name="List of economies", skiprows=3, header=None, names=column_names)
    income_df = income_df[["Country Name", "Country Code", "Income Level"]]
    print("✅ Successfully loaded World Bank income classifications")
except:
    print("⚠️ Could not load online classifications, using manual mapping")
    # Fallback manual classification (you can expand this)
    income_df = pd.DataFrame({
        'Country Name': ['Afghanistan', 'Bangladesh', 'India', 'Brazil', 'China'],
        'Country Code': ['AFG', 'BGD', 'IND', 'BRA', 'CHN'],
        'Income Level': ['Low income', 'Lower middle income', 'Lower middle income', 
                        'Upper middle income', 'Upper middle income']
    })

# Merge with income classifications
df = df.merge(income_df, on=["Country Name", "Country Code"], how="left")

print(f"Income level distribution before filtering:")
print(df["Income Level"].value_counts())

# ============================================================================
# STEP 7: ENHANCED SAMPLE FILTERING
# ============================================================================

print("\n🔍 Step 7: Enhanced sample filtering...")

# Remove regional aggregates and non-country entities (comprehensive list)
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

df = df[~df["Country Name"].isin(regions_to_remove)]

# Enhanced manual income level corrections with documentation
manual_income_levels = {
    "Cote d'Ivoire": "Lower middle income",
    "Sao Tome and Principe": "Lower middle income", 
    "Turkiye": "Upper middle income",
    "Viet Nam": "Lower middle income",
    "Afghanistan": "Low income",
    "Venezuela, RB": "Not classified",  # Economic crisis makes classification difficult
    "Aruba": "High income",
    "Curacao": "High income",
    "Czechia": "High income"  # Updated classification
}

# Apply manual corrections
df["Income Level"] = df["Country Name"].map(manual_income_levels).fillna(df["Income Level"])

# Remove high-income and unclassified countries (focus on development economics)
df = df[~df["Income Level"].isin(["High income", "Not classified"])]

# Remove countries that recently graduated to high income (2024-2025 World Bank update)
recently_graduated = ["Bulgaria", "Palau", "Russian Federation"]
df = df[~df["Country Name"].isin(recently_graduated)]

print(f"Final income level distribution:")
print(df["Income Level"].value_counts())
print(f"Final country count: {df['Country Name'].nunique()}")

# ============================================================================
# STEP 8: ENHANCED DATA AVAILABILITY ASSESSMENT
# ============================================================================

print("\n📊 Step 8: Enhanced data availability assessment...")

# Assess data availability by country-series combination
availability_stats = {}

for income_level in df["Income Level"].unique():
    if pd.notna(income_level):
        subset = df[df["Income Level"] == income_level]
        
        # Calculate completion rate
        total_possible = len(subset) * len(year_columns)
        non_missing = subset[year_columns].count().sum()
        completion_rate = (non_missing / total_possible) * 100
        
        availability_stats[income_level] = {
            'countries': subset['Country Name'].nunique(),
            'series': subset['Series Name'].nunique(),
            'completion_rate': completion_rate
        }
        
        print(f"{income_level}: {subset['Country Name'].nunique()} countries, "
              f"{subset['Series Name'].nunique()} series, "
              f"{completion_rate:.1f}% data completion")

# Enhanced missing data threshold (series-specific)
print("\nApplying enhanced missing data thresholds...")

# Instead of arbitrary threshold, drop country-series with <30% data availability
min_data_points = int(0.3 * len(year_columns))  # At least 30% of years must have data

before_drop = len(df)
df = df.dropna(thresh=len(non_year_columns) + min_data_points, axis=0)
after_drop = len(df)

print(f"Dropped {before_drop - after_drop:,} observations with insufficient data")
print(f"Retained {after_drop:,} observations ({(after_drop/before_drop)*100:.1f}%)")

# ============================================================================
# STEP 9: FINAL VALIDATION AND QUALITY CHECKS
# ============================================================================

print("\n✅ Step 9: Final validation and quality checks...")

# Check for duplicates
duplicates = df.duplicated()
if duplicates.sum() > 0:
    print(f"Warning: Found {duplicates.sum()} duplicate rows - removing them")
    df = df.drop_duplicates()
else:
    print("✅ No duplicate rows found")

# Validate income classifications
valid_income_levels = ["Low income", "Lower middle income", "Upper middle income"]
invalid_income = df[~df["Income Level"].isin(valid_income_levels)]["Income Level"].unique()
if len(invalid_income) > 0:
    print(f"Warning: Invalid income levels found: {invalid_income}")
    df = df[df["Income Level"].isin(valid_income_levels)]
else:
    print("✅ All income classifications are valid")

# Final data quality summary
print("\n" + "="*60)
print("📋 FINAL DATA QUALITY SUMMARY")
print("="*60)

print(f"Final dataset shape: {df.shape}")
print(f"Countries: {df['Country Name'].nunique()}")
print(f"Series indicators: {df['Series Name'].nunique()}")
print(f"Time span: {len(year_columns)} years")

# Missing data summary
total_cells = len(df) * len(year_columns)
missing_cells = df[year_columns].isnull().sum().sum()
completion_rate = ((total_cells - missing_cells) / total_cells) * 100

print(f"Overall data completion rate: {completion_rate:.1f}%")

# ============================================================================
# STEP 10: SAVE ENHANCED DATASET
# ============================================================================

print("\n💾 Step 10: Saving enhanced dataset...")

# Reset index for clean saving
df.reset_index(drop=True, inplace=True)

# Save with detailed metadata
output_path = '../datasets/enhanced_growth_rates_emissions_energy_prod_income_level_country_df.csv'
df.to_csv(output_path, index=False)

# Create metadata file
metadata = {
    'processing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_observations': len(df),
    'unique_countries': df['Country Name'].nunique(),
    'unique_series': df['Series Name'].nunique(),
    'time_span': f"{year_columns[0]} to {year_columns[-1]}",
    'data_completion_rate': f"{completion_rate:.1f}%",
    'income_distribution': df['Income Level'].value_counts().to_dict(),
    'outliers_winsorized': total_outliers,
    'validation_issues_corrected': validation_issues,
    'processing_methods': [
        'Within-group temporal imputation only',
        'IQR-based outlier winsorization (5th-95th percentile)',
        'Statistical validation rules applied',
        'Enhanced country classification',
        'Minimum 30% data availability threshold'
    ]
}

metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv('../datasets/preprocessing_metadata.csv', index=False)

print(f"✅ Enhanced dataset saved: {output_path}")
print(f"✅ Metadata saved: ../datasets/preprocessing_metadata.csv")

print("\n🎉 ENHANCED PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*60)
print("Key improvements over original preprocessing:")
print("• Removed problematic cross-series imputation")
print("• Added statistical outlier detection and winsorization")
print("• Enhanced data quality validation")
print("• Improved missing data thresholds")
print("• Added comprehensive documentation and metadata")
print("• Follows econometric best practices for panel data")

# ============================================================================
# OPTIONAL: GENERATE PREPROCESSING REPORT
# ============================================================================

def generate_preprocessing_report():
    """Generate a comprehensive preprocessing report"""
    
    report = f"""
# Enhanced Preprocessing Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Summary
- **Total Observations**: {len(df):,}
- **Countries**: {df['Country Name'].nunique()}
- **Series Indicators**: {df['Series Name'].nunique()}
- **Time Coverage**: {year_columns[0]} to {year_columns[-1]}
- **Data Completion**: {completion_rate:.1f}%

## Income Distribution
{df['Income Level'].value_counts().to_string()}

## Processing Steps Applied
1. [DONE] Within-group temporal imputation
2. [DONE] Statistical outlier detection (IQR method)
3. [DONE] Data validation rules
4. [DONE] Enhanced country classification
5. [DONE] Minimum data availability thresholds
6. [DONE] Comprehensive quality checks

## Quality Improvements
- **Outliers winsorized**: {total_outliers}
- **Validation issues corrected**: {validation_issues}
- **Completion rate**: {completion_rate:.1f}%
- **No spurious cross-series imputation**

## Ready for GMM Analysis
This dataset follows econometric best practices and is suitable for:
- Panel data analysis
- Dynamic GMM estimation
- IV/2SLS regression
- Fixed effects models
"""
    
    # Fix: Specify UTF-8 encoding explicitly
    with open('../datasets/preprocessing_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Report saved: ../datasets/preprocessing_report.md")

# Generate report
generate_preprocessing_report()

print("\n🚀 Dataset ready for GMM analysis!")