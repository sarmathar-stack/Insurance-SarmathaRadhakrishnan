# Insurance Claims Bias Analysis Dashboard

A comprehensive Streamlit dashboard for detecting bias in insurance claim settlements.

## Features

- **Descriptive Analytics**: Cross-tabulation analysis across 7 dimensions
- **Diagnostic Analysis**: Deep-dive bias detection (Zone, Age, Income, Sum Assured)
- **Predictive Modeling**: KNN, Decision Tree, Random Forest, Gradient Boosting
- **Model Comparison**: ROC curves, confusion matrices, feature importance
- **Findings & Recommendations**: Actionable insights with priority-based action plan

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `Insurance.csv` | Dataset (upload your own) |
| `requirements.txt` | Python dependencies |
| `fig1_crosstab_heatmaps.png` | Cross-tabulation visualizations |
| `fig2_diagnostic_analysis.png` | Diagnostic analysis charts |
| `fig3_model_performance.png` | Model performance comparison |

## Deployment

1. Upload all files to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and deploy

## Key Findings

- **Payment Mode Bias**: Quarterly (55.0%) vs Single (10.1%)
- **Income Bias**: Low income (51.9%) vs Very High (22.4%)
- **Zone Disparity**: PENINSULAR (76.9%) vs JKB CREDITOR (3.4%)
- **Best Model**: Random Forest (73.4% accuracy, AUC 0.744)
