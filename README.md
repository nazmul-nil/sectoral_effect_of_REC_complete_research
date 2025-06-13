```markdown
# Sectoral Impact of Renewable Energy Consumption on Economic Transformation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/research-econometrics-green.svg)](https://github.com/nazmul-nil/sectoral_effect_of_REC_complete_research)

## 🔬 Project Overview

This repository provides a comprehensive econometric analysis examining **how renewable energy consumption (REC) and energy efficiency transitions reshape sectoral economic transformation trajectories** across low- and middle-income countries. Using panel data spanning 1990-2023 and covering 128 countries, the study employs advanced econometric techniques including Fixed Effects, Instrumental Variables (IV/2SLS), and Generalized Method of Moments (GMM) to identify causal relationships and heterogeneous pathways.

**This work is a continuation of the MSc thesis,** *['Investigating the Effects of Renewable Energy and Energy Efficiency on Economic Growth in Low- and Middle-Income Nations'](https://github.com/nazmul-nil/MSc-DS-Research-Project-UH.git)*

### 🎯 Key Research Questions
- How do renewable energy adoption pathways differ across income levels and development stages?
- What are the threshold effects governing renewable energy impacts on sectoral growth?
- How have these relationships evolved over three decades of energy transitions?

### 🏆 Key Findings
- **Income-level heterogeneity**: Renewable energy effects vary dramatically between low-income and middle-income countries
- **Threshold effects**: Non-linear relationships with critical consumption levels at 9.7% and 32.9% renewable energy shares
- **Temporal dynamics**: Evolving energy-growth relationships from 1990s to 2020s
- **Sectoral differentiation**: Distinct impacts across agriculture, industry, and services sectors

---

## 📂 Repository Structure

```
📦 sectoral_effect_of_REC_complete_research/
├── 📁 datasets/                          # Data files and documentation
│   ├── 📁 P_Data_Extract_From_World_Development_Indicators/  # Original WDI data
│   ├── 📄 enhanced_growth_rates_emissions_energy_prod_income_level_country_df.csv
│   ├── 📄 growth_rates_energy_gmm_base.csv
│   ├── 📄 growth_rates_energy_gmm_ready.csv
│   └── 📄 preprocessing_metadata.csv
├── 📁 notebooks/                         # Analysis workflow (Jupyter notebooks)
│   ├── 📓 data_preprocessing.ipynb       # 1️⃣ Data cleaning & preparation
│   ├── 📓 00_prepare_gmm_base_dataset.ipynb  # 2️⃣ Panel structure creation
│   ├── 📓 01_create_lagged_variables.ipynb   # 3️⃣ Lag generation for GMM
│   ├── 📓 02_system_gmm_agriculture.ipynb    # 4️⃣ Main econometric analysis
│   ├── 📓 03_full_gmm_agriculture_analysis.ipynb  # 5️⃣ Robustness checks
│   └── 📓 04_visualizations_predictions.ipynb     # 6️⃣ Publication-ready plots
├── 📁 plots/                            # Generated visualizations
│   ├── 🖼️ income_heterogeneity_effects.png
│   ├── 🖼️ threshold_effects_analysis.png
│   ├── 🖼️ temporal_dynamics_evolution.png
│   └── 🖼️ model_performance_comparison.png
├── 📁 results/                          # Saved model outputs
│   ├── 📊 master_gmm_results.pkl
│   ├── 📊 robustness_results.pkl
│   └── 📊 coefficient_comparison.pkl
├── 📁 python_script_file/              # Utility scripts
├── 📄 requirements.txt                  # Python dependencies
├── 📄 LICENSE                          # MIT License
├── 📄 commit_script.sh                 # Git automation
└── 📄 README.md                        # This file
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook/Lab
- Git (for cloning)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nazmul-nil/sectoral_effect_of_REC_complete_research.git
   cd sectoral_effect_of_REC_complete_research
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

### 📋 Execution Workflow

**⚠️ Important**: Run notebooks in the specified order for reproducibility

| Step | Notebook | Purpose | Duration |
|------|----------|---------|----------|
| 1️⃣ | `data_preprocessing.ipynb` | Enhanced data cleaning, outlier treatment, income classification | ~10 min |
| 2️⃣ | `00_prepare_gmm_base_dataset.ipynb` | Reshape to panel format, select 8 key indicators | ~5 min |
| 3️⃣ | `01_create_lagged_variables.ipynb` | Generate 1-3 period lags and first differences | ~3 min |
| 4️⃣ | `02_system_gmm_agriculture.ipynb` | Main GMM estimations with heterogeneity analysis | ~15 min |
| 5️⃣ | `03_full_gmm_agriculture_analysis.ipynb` | Robustness checks (FD, Dynamic Panel, DK errors) | ~20 min |
| 6️⃣ | `04_visualizations_predictions.ipynb` | Generate publication-ready visualizations | ~8 min |

### 🎯 Expected Outputs

After running all notebooks, you will have:
- ✅ Cleaned and analysis-ready datasets
- ✅ Complete econometric estimation results
- ✅ Robustness check validations
- ✅ Professional publication-ready visualizations
- ✅ Comprehensive model diagnostics

---

## 📊 Data Description

### Primary Data Source
- **World Bank World Development Indicators (WDI)**
- **Coverage**: 128 countries, 1990-2023
- **Original scope**: All countries, 91 indicators, 1989-2023

### Key Variables
| Variable | Description | Source Code |
|----------|-------------|-------------|
| **REC** | Renewable electricity output (% of total) | `EG.ELC.RNEW.ZS` |
| **EI** | Energy intensity (kg oil eq. per $1,000 GDP) | `EG.USE.COMM.GD.PP.KD` |
| **AgriGrowth** | Agriculture value added (annual % growth) | `NV.AGR.TOTL.KD.ZG` |
| **IndGrowth** | Industry value added (annual % growth) | `NV.IND.TOTL.KD.ZG` |
| **ServGrowth** | Services value added (annual % growth) | `NV.SRV.TOTL.KD.ZG` |
| **AccessElec** | Access to electricity (% of population) | `EG.ELC.ACCS.ZS` |
| **PM2.5** | Air pollution exposure (μg/m³) | `EN.ATM.PM25.MC.M3` |

### Income Classifications
- **Low income**: 18.3% of sample (n=731)
- **Lower middle income**: 41.0% of sample (n=1,634)  
- **Upper middle income**: 38.8% of sample (n=1,498)

> 📎 **Download Original Data**: Access the complete WDI dataset at [World Bank DataBank](https://databank.worldbank.org/source/world-development-indicators)

---

## 🔬 Methodology Overview

### Econometric Techniques
- **Fixed Effects Panel Models**: Control for country-specific characteristics
- **Instrumental Variables (IV/2SLS)**: Address endogeneity concerns
- **System GMM**: Dynamic panel estimation with lagged instruments
- **Threshold Analysis**: Identify non-linear relationships
- **Heterogeneity Analysis**: Income-group-specific estimations

### Robustness Checks
- ✅ First Differences specification
- ✅ Dynamic Panel models with lagged dependent variables
- ✅ Post-2000 subsample analysis
- ✅ Driscoll-Kraay standard errors
- ✅ Alternative instrument sets

### Key Innovations
- **Comprehensive heterogeneity analysis** across income levels
- **Threshold effect identification** with regime-specific impacts
- **Temporal dynamics assessment** spanning three decades
- **Multi-sector analysis** covering agriculture, industry, and services

---

## 📈 Key Results & Visualizations

### Main Findings

1. **Income-Level Heterogeneity**
   - Low-income countries: Negative industrial effects (-0.128, p<0.01)
   - Upper middle-income: Positive industrial effects (0.053, p<0.10)
   - Policy implication: Development-stage-specific strategies needed

2. **Threshold Effects**
   - Critical thresholds at 9.7% and 32.9% renewable energy shares
   - Non-linear relationships across all sectors
   - Optimal renewable energy "sweet spots" identified

3. **Temporal Evolution**
   - Services sector: -0.019 (1990s) → 0.001 (2020s)
   - Industrial sector: Persistent negative effects over time
   - Evidence of learning effects and technological advancement

### Generated Visualizations
- 🎨 Income-level heterogeneity comparison plots
- 📊 Threshold effects heatmaps and regime analysis
- 📈 Temporal dynamics evolution charts
- 🔍 Model performance and diagnostic plots
- 🌍 Cross-country scatter pattern analysis

---

## 💡 Policy Implications

### For Policymakers
- **Low-income countries**: Prioritize complementary infrastructure investments
- **Middle-income countries**: Leverage renewable energy for industrial competitiveness  
- **All countries**: Consider threshold effects in energy transition planning

### For Researchers
- Framework for analyzing heterogeneous energy transition pathways
- Methodology for threshold identification in development economics
- Template for multi-dimensional robustness checking

### For Development Organizations
- Evidence base for differentiated climate finance strategies
- Guidance for country-specific renewable energy programs
- Insights for sustainable development goal implementation

---

## 📚 Citation & Academic Use

### How to Cite This Work

**For Academic Papers:**
```bibtex
@misc{hossain2025sectoral,
  author       = {Hossain, Nazmul},
  title        = {Sectoral Impact of Renewable Energy Consumption on Economic Transformation: 
                  Evidence from Low- and Middle-Income Countries},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/nazmul-nil/sectoral_effect_of_REC_complete_research}},
  note         = {Accessed: [DATE]}
}
```

**For Reports/Policy Briefs:**
> Hossain, N. (2025). Sectoral Impact of Renewable Energy Consumption on Economic Transformation. 
> Available at: https://github.com/nazmul-nil/sectoral_effect_of_REC_complete_research

### Related Publications
- 📄 Working paper version (forthcoming)
- 📊 Policy brief series (in preparation)
- 🎯 Sector-specific deep dives (planned)

---

## 🤝 Contributing & Collaboration

### Ways to Contribute
- 🐛 **Bug Reports**: Open an issue for any errors or problems
- 💡 **Feature Requests**: Suggest improvements or extensions
- 🔧 **Code Contributions**: Submit pull requests with enhancements
- 📊 **Data Updates**: Help extend the analysis to newer periods
- 📝 **Documentation**: Improve clarity and completeness

### Collaboration Opportunities
- Extension to firm-level analysis
- Integration of additional environmental indicators
- Subnational analysis for federal countries
- Technology-specific renewable energy disaggregation

### Getting Help
- 📧 Open a GitHub issue for technical questions
- 💬 Use discussion forums for methodology clarifications
- 📖 Check existing issues before posting new questions

---

## 🔐 License & Terms

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Terms of Use
- ✅ **Free use** for academic and research purposes
- ✅ **Modification and distribution** permitted with attribution
- ✅ **Commercial use** allowed with proper citation
- ❌ **No warranty** provided - use at your own risk

### Copyright Notice
© 2025 Nazmul Hossain. All rights reserved.

---

## 📞 Contact & Support

### Project Maintainer
**Nazmul Hossain**
- 🔗 GitHub: [@nazmul-nil](https://github.com/nazmul-nil)
- 📧 Contact: Via GitHub issues or repository discussions

### Project Links
- 🏠 **Repository**: [sectoral_effect_of_REC_complete_research](https://github.com/nazmul-nil/sectoral_effect_of_REC_complete_research)
- 📊 **Data Source**: [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators)
- 📄 **Documentation**: See individual notebook headers and comments

---

## 🏷️ Keywords

`renewable energy` • `economic development` • `panel econometrics` • `GMM estimation` • `energy transitions` • `sectoral analysis` • `development economics` • `sustainability` • `threshold effects` • `heterogeneous pathways`

---

*Last updated: January 2025*
```

### 🌟 Key Enhancements Made:

1. **Professional badges** and visual hierarchy
2. **Comprehensive structure** with clear navigation
3. **Detailed methodology** overview and key innovations
4. **Policy implications** section for broader impact
5. **Academic citation** formats for proper attribution
6. **Collaboration guidelines** to encourage community engagement
7. **Visual organization** with emojis and tables for better readability
8. **Comprehensive contact** and support information
9. **Keywords section** for discoverability
10. **Professional formatting** following open-source project standards
