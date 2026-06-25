"""
Insurance Claims Settlement Bias Analysis Dashboard
====================================================
A comprehensive analytics dashboard for detecting bias in insurance claim settlements.
Features: Descriptive Analytics, Diagnostic Analysis, Predictive Modeling (KNN, Decision Tree, 
Random Forest, Gradient Boosting), and Interactive Visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Insurance Claims Bias Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e5984;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

@st.cache_data
def load_data():
    """Load and preprocess the insurance claims data."""
    df = pd.read_csv('Insurance.csv')

    # Clean numeric columns
    df['SUM_ASSURED'] = df['SUM_ASSURED'].str.replace(',', '').astype(float)
    df['PI_ANNUAL_INCOME'] = df['PI_ANNUAL_INCOME'].str.replace(',', '').astype(float)

    # Fill missing values
    df['REASON_FOR_CLAIM'] = df['REASON_FOR_CLAIM'].fillna('Unknown')
    df['PI_OCCUPATION'] = df['PI_OCCUPATION'].fillna('Unknown')

    # Create target variable
    df['TARGET'] = (df['POLICY_STATUS'] == 'Repudiate Death').astype(int)

    # Feature Engineering
    df['AGE_GROUP'] = pd.cut(df['PI_AGE'], bins=[0, 30, 40, 50, 60, 70, 100],
                              labels=['<30', '30-40', '40-50', '50-60', '60-70', '70+'])

    df['INCOME_GROUP'] = pd.cut(df['PI_ANNUAL_INCOME'],
                                 bins=[-1, 0, 100000, 300000, 500000, 10000000],
                                 labels=['Zero Income', 'Low (1-1L)', 'Medium (1L-3L)', 
                                        'High (3L-5L)', 'Very High (5L+)'])

    df['SA_GROUP'] = pd.cut(df['SUM_ASSURED'],
                             bins=[-1, 100000, 300000, 500000, 1000000, 20000000],
                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    def categorize_zone(zone):
        zone = str(zone).upper()
        if 'TEAM' in zone:
            return 'TEAM'
        elif any(x in zone for x in ['JKB', 'KBL', 'PNB', 'RRB', 'CREDITOR']):
            return 'BANK_CHANNEL'
        elif zone == 'AGENCY':
            return 'AGENCY'
        elif zone in ['NORTH', 'SOUTH', 'EAST', 'WEST', 'NORTH 1', 'SOUTH 1', 'SOUTH 2', 'EAST 1', 'EAST 2']:
            return 'REGIONAL'
        else:
            return 'OTHER'

    df['ZONE_CATEGORY'] = df['ZONE'].apply(categorize_zone)

    return df

# ============================================
# MACHINE LEARNING PIPELINE
# ============================================

@st.cache_resource
def train_models(df):
    """Train all classification models."""
    feature_cols = ['PI_GENDER', 'SUM_ASSURED', 'PAYMENT_MODE', 'EARLY_NON',
                    'MEDICAL_NONMED', 'PI_AGE', 'PI_ANNUAL_INCOME',
                    'AGE_GROUP', 'INCOME_GROUP', 'SA_GROUP', 'ZONE_CATEGORY']

    X = df[feature_cols].copy()
    y = df['TARGET'].copy()

    categorical_cols = ['PI_GENDER', 'PAYMENT_MODE', 'EARLY_NON', 'MEDICAL_NONMED',
                        'AGE_GROUP', 'INCOME_GROUP', 'SA_GROUP', 'ZONE_CATEGORY']
    numerical_cols = ['PI_AGE', 'SUM_ASSURED', 'PI_ANNUAL_INCOME']

    # Encode categorical
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    # Scale numerical
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # Models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=20),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, 
                                                 max_depth=15, min_samples_split=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42,
                                                         max_depth=5, learning_rate=0.1)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_test_prob = model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)

        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_test_pred)

        results[name] = {
            'train_acc': train_acc, 'test_acc': test_acc,
            'precision': precision, 'recall': recall, 'f1': f1,
            'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'cm': cm,
            'y_test': y_test, 'y_test_pred': y_test_pred, 'y_test_prob': y_test_prob
        }

    # Feature importance
    rf_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': trained_models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)

    gb_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': trained_models['Gradient Boosting'].feature_importances_
    }).sort_values('Importance', ascending=False)

    return results, rf_imp, gb_imp, X_test, y_test

# ============================================
# SIDEBAR
# ============================================

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Select Analysis Module:", [
    "🏠 Home / Overview",
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analysis",
    "🤖 Predictive Modeling",
    "📈 Model Comparison",
    "📋 Findings & Recommendations"
])

st.sidebar.markdown("---")
st.sidebar.info("""
**About This Dashboard**
This tool analyzes insurance claim settlement data to detect potential biases using:
- Descriptive Statistics
- Diagnostic Analytics
- Machine Learning (KNN, DT, RF, GB)
""")

# ============================================
# MAIN CONTENT
# ============================================

# Load data
try:
    df = load_data()
    results, rf_imp, gb_imp, X_test, y_test = train_models(df)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure 'Insurance.csv' is in the same directory as app.py")
    data_loaded = False

if data_loaded:
    total_claims = len(df)
    approved = (df['POLICY_STATUS'] == 'Approved Death Claim').sum()
    repudiated = (df['POLICY_STATUS'] == 'Repudiate Death').sum()
    repudiation_rate = (repudiated / total_claims) * 100

    # ============================================
    # HOME PAGE
    # ============================================
    if page == "🏠 Home / Overview":
        st.markdown('<div class="main-header">🏛️ Insurance Claims Settlement Bias Analysis</div>', 
                    unsafe_allow_html=True)

        st.markdown("""
        Welcome to the **Insurance Claims Bias Analysis Dashboard**. This comprehensive tool helps 
        identify potential biases in claim settlement processes through descriptive, diagnostic, 
        and predictive analytics.
        """)

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", f"{total_claims:,}")
        with col2:
            st.metric("Approved", f"{approved:,}", f"{(approved/total_claims)*100:.1f}%")
        with col3:
            st.metric("Repudiated", f"{repudiated:,}", f"{repudiation_rate:.1f}%", delta_color="inverse")
        with col4:
            st.metric("Avg Age", f"{df['PI_AGE'].mean():.1f} years")

        st.markdown("---")

        # Dataset Overview
        st.markdown('<div class="sub-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Columns:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
            st.write("**Duplicate Rows:**", df.duplicated().sum())

        st.markdown("---")

        # Quick Summary Stats
        st.markdown('<div class="sub-header">📈 Quick Statistics</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Age Distribution**")
            st.write(df['PI_AGE'].describe())
        with col2:
            st.write("**Sum Assured Distribution**")
            st.write(df['SUM_ASSURED'].describe())
        with col3:
            st.write("**Annual Income Distribution**")
            st.write(df['PI_ANNUAL_INCOME'].describe())

        # Sample Data
        st.markdown('<div class="sub-header">🔍 Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

    # ============================================
    # DESCRIPTIVE ANALYTICS
    # ============================================
    elif page == "📊 Descriptive Analytics":
        st.markdown('<div class="main-header">📊 Descriptive Analytics: Cross-Tabulation Analysis</div>', 
                    unsafe_allow_html=True)

        st.markdown("""
        This section provides cross-tabulation analysis of various features against the policy status 
        to understand the distribution and basic relationships in the data.
        """)

        # Cross-tabulation selector
        analysis_type = st.selectbox("Select Cross-Tabulation Analysis:", [
            "Gender vs Policy Status",
            "Early/Non-Early vs Policy Status",
            "Medical/Non-Medical vs Policy Status",
            "Age Group vs Policy Status",
            "Payment Mode vs Policy Status",
            "Income Group vs Policy Status",
            "Zone Category vs Policy Status"
        ])

        if analysis_type == "Gender vs Policy Status":
            crosstab = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['PI_GENDER'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            # Visualization
            fig = px.bar(crosstab_pct.reset_index(), x='PI_GENDER', 
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Gender vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71', 
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="insight-box">💡 <b>Insight:</b> Males have a slightly higher '
                       'repudiation rate (32.7%) compared to females (28.6%).</div>', 
                       unsafe_allow_html=True)

        elif analysis_type == "Early/Non-Early vs Policy Status":
            crosstab = pd.crosstab(df['EARLY_NON'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['EARLY_NON'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='EARLY_NON',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Early/Non-Early vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="warning-box">⚠️ <b>Bias Alert:</b> Non-Early claims have a '
                       'significantly higher repudiation rate (37.2%) compared to Early claims (23.0%). '
                       'This suggests potential bias against non-early claimants.</div>',
                       unsafe_allow_html=True)

        elif analysis_type == "Medical/Non-Medical vs Policy Status":
            crosstab = pd.crosstab(df['MEDICAL_NONMED'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['MEDICAL_NONMED'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='MEDICAL_NONMED',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Medical/Non-Medical vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="success-box">✅ <b>Observation:</b> Medical cases have a lower '
                       'repudiation rate (18.9%) compared to Non-Medical cases (33.6%), indicating '
                       'better documentation or clearer claim validity in medical cases.</div>',
                       unsafe_allow_html=True)

        elif analysis_type == "Age Group vs Policy Status":
            crosstab = pd.crosstab(df['AGE_GROUP'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['AGE_GROUP'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='AGE_GROUP',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Age Group vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="insight-box">💡 <b>Insight:</b> The 30-40 age group shows the '
                       'highest repudiation rate (36.7%), while very young (<30) and very old (70+) '
                       'groups have lower rates.</div>', unsafe_allow_html=True)

        elif analysis_type == "Payment Mode vs Policy Status":
            crosstab = pd.crosstab(df['PAYMENT_MODE'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['PAYMENT_MODE'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='PAYMENT_MODE',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Payment Mode vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="warning-box">⚠️ <b>Major Bias Detected:</b> Quarterly payment '
                       'mode has an alarmingly high repudiation rate of 55.0%, while Single payment '
                       'mode has only 10.1%. This is a strong indicator of systematic bias.</div>',
                       unsafe_allow_html=True)

        elif analysis_type == "Income Group vs Policy Status":
            crosstab = pd.crosstab(df['INCOME_GROUP'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['INCOME_GROUP'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='INCOME_GROUP',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Income Group vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="warning-box">⚠️ <b>Income Bias:</b> Low income group (1-1L) has '
                       'a repudiation rate of 51.9%, nearly double the overall average. Higher income '
                       'groups show progressively lower repudiation rates.</div>',
                       unsafe_allow_html=True)

        elif analysis_type == "Zone Category vs Policy Status":
            crosstab = pd.crosstab(df['ZONE_CATEGORY'], df['POLICY_STATUS'], margins=True)
            crosstab_pct = pd.crosstab(df['ZONE_CATEGORY'], df['POLICY_STATUS'], normalize='index') * 100

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Count Table**")
                st.dataframe(crosstab, use_container_width=True)
            with col2:
                st.write("**Percentage Table**")
                st.dataframe(crosstab_pct.round(2), use_container_width=True)

            fig = px.bar(crosstab_pct.reset_index(), x='ZONE_CATEGORY',
                        y=['Approved Death Claim', 'Repudiate Death'],
                        barmode='group', title='Zone Category vs Policy Status (%)',
                        color_discrete_map={'Approved Death Claim': '#2ecc71',
                                           'Repudiate Death': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="insight-box">💡 <b>Zone Insight:</b> AGENCY channel has the '
                       'highest repudiation rate among major categories, while BANK_CHANNEL shows '
                       'lower rates.</div>', unsafe_allow_html=True)

    # ============================================
    # DIAGNOSTIC ANALYSIS
    # ============================================
    elif page == "🔍 Diagnostic Analysis":
        st.markdown('<div class="main-header">🔍 Diagnostic Analysis: Bias Detection</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        This section performs deep-dive diagnostic analysis to probe potential biased behavior 
        in claim settlements across different dimensions.
        """)

        diag_type = st.selectbox("Select Diagnostic Dimension:", [
            "Zone/Team-wise Bias",
            "Age-wise Bias",
            "Income-wise Bias",
            "Sum Assured-wise Bias",
            "Reason for Claim Analysis",
            "Combined Bias Heatmap"
        ])

        if diag_type == "Zone/Team-wise Bias":
            zone_stats = df.groupby('ZONE').agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100,
                'POLICY_NO': 'count'
            }).reset_index()
            zone_stats.columns = ['ZONE', 'Repudiation_Rate', 'Total_Cases']
            zone_stats = zone_stats[zone_stats['Total_Cases'] >= 5]  # Statistical significance
            zone_stats = zone_stats.sort_values('Repudiation_Rate', ascending=False)

            st.write("**Zone-wise Repudiation Analysis (Zones with ≥5 cases)**")
            st.dataframe(zone_stats.round(2), use_container_width=True)

            # Color coding
            zone_stats['Color'] = zone_stats['Repudiation_Rate'].apply(
                lambda x: 'red' if x > 45 else 'orange' if x > 32 else 'green')

            fig = px.bar(zone_stats, x='Repudiation_Rate', y='ZONE', orientation='h',
                        color='Color', color_discrete_map={'red': '#e74c3c', 
                                                          'orange': '#f39c12', 
                                                          'green': '#2ecc71'},
                        title='Zone-wise Repudiation Rate',
                        labels={'Repudiation_Rate': 'Repudiation Rate (%)'})
            fig.add_vline(x=repudiation_rate, line_dash="dash", line_color="black",
                         annotation_text=f"Overall Avg: {repudiation_rate:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

            high_bias_zones = zone_stats[zone_stats['Repudiation_Rate'] > 45]['ZONE'].tolist()
            if high_bias_zones:
                st.markdown(f'<div class="warning-box">🚨 <b>High Bias Zones:</b> '
                           f'{", ".join(high_bias_zones)} have repudiation rates >45%.</div>',
                           unsafe_allow_html=True)

        elif diag_type == "Age-wise Bias":
            age_stats = df.groupby('AGE_GROUP').agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100,
                'POLICY_NO': 'count'
            }).reset_index()
            age_stats.columns = ['AGE_GROUP', 'Repudiation_Rate', 'Total_Cases']

            st.write("**Age-wise Repudiation Analysis**")
            st.dataframe(age_stats.round(2), use_container_width=True)

            fig = px.line(age_stats, x='AGE_GROUP', y='Repudiation_Rate', 
                         markers=True, title='Age-wise Repudiation Trend',
                         labels={'Repudiation_Rate': 'Repudiation Rate (%)'})
            fig.add_hline(y=repudiation_rate, line_dash="dash", line_color="red",
                         annotation_text=f"Overall Avg: {repudiation_rate:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="insight-box">💡 The 30-40 age group shows peak repudiation, '
                       'possibly due to higher scrutiny on younger claimants or more complex cases.</div>',
                       unsafe_allow_html=True)

        elif diag_type == "Income-wise Bias":
            income_stats = df.groupby('INCOME_GROUP').agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100,
                'POLICY_NO': 'count'
            }).reset_index()
            income_stats.columns = ['INCOME_GROUP', 'Repudiation_Rate', 'Total_Cases']

            st.write("**Income-wise Repudiation Analysis**")
            st.dataframe(income_stats.round(2), use_container_width=True)

            fig = px.bar(income_stats, x='INCOME_GROUP', y='Repudiation_Rate',
                        color='Repudiation_Rate', color_continuous_scale='RdYlGn_r',
                        title='Income-wise Repudiation Rate',
                        labels={'Repudiation_Rate': 'Repudiation Rate (%)'})
            fig.add_hline(y=repudiation_rate, line_dash="dash", line_color="red",
                         annotation_text=f"Overall Avg: {repudiation_rate:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="warning-box">⚠️ <b>Strong Income Bias:</b> Low income claimants '
                       '(1-1L) face 51.9% repudiation vs 22.4% for very high income. This suggests '
                       'potential socioeconomic discrimination.</div>', unsafe_allow_html=True)

        elif diag_type == "Sum Assured-wise Bias":
            df['SA_BIN'] = pd.cut(df['SUM_ASSURED'], bins=10)
            sa_stats = df.groupby('SA_BIN').agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100,
                'POLICY_NO': 'count'
            }).reset_index()
            sa_stats.columns = ['SA_RANGE', 'Repudiation_Rate', 'Total_Cases']

            sa_labels = [f"{int(interval.left/1000)}K-{int(interval.right/1000)}K" 
                        for interval in sa_stats['SA_RANGE']]
            sa_stats['SA_LABEL'] = sa_labels

            st.write("**Sum Assured-wise Repudiation Analysis**")
            st.dataframe(sa_stats[['SA_LABEL', 'Repudiation_Rate', 'Total_Cases']].round(2), 
                        use_container_width=True)

            fig = px.bar(sa_stats, x='SA_LABEL', y='Repudiation_Rate',
                        color='Repudiation_Rate', color_continuous_scale='RdYlGn_r',
                        title='Sum Assured-wise Repudiation Rate')
            fig.add_hline(y=repudiation_rate, line_dash="dash", line_color="red",
                         annotation_text=f"Overall Avg: {repudiation_rate:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

        elif diag_type == "Reason for Claim Analysis":
            reason_df = df[df['REASON_FOR_CLAIM'] != 'Unknown']
            reason_stats = reason_df.groupby('REASON_FOR_CLAIM').agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100,
                'POLICY_NO': 'count'
            }).reset_index()
            reason_stats.columns = ['REASON', 'Repudiation_Rate', 'Total_Cases']
            reason_stats = reason_stats[reason_stats['Total_Cases'] >= 5]
            reason_stats = reason_stats.sort_values('Repudiation_Rate', ascending=False)

            st.write("**Reason for Claim-wise Repudiation Analysis (≥5 cases)**")
            st.dataframe(reason_stats.round(2), use_container_width=True)

            fig = px.bar(reason_stats.head(15), x='Repudiation_Rate', y='REASON',
                        orientation='h', color='Repudiation_Rate',
                        color_continuous_scale='RdYlGn_r',
                        title='Top 15 Reasons by Repudiation Rate')
            fig.add_vline(x=repudiation_rate, line_dash="dash", line_color="red",
                         annotation_text=f"Overall Avg: {repudiation_rate:.1f}%")
            st.plotly_chart(fig, use_container_width=True)

        elif diag_type == "Combined Bias Heatmap":
            st.write("**Multi-dimensional Bias Heatmap**")

            # Create a pivot table for heatmap
            heatmap_data = df.groupby(['AGE_GROUP', 'INCOME_GROUP']).agg({
                'POLICY_STATUS': lambda x: (x == 'Repudiate Death').mean() * 100
            }).reset_index()
            heatmap_pivot = heatmap_data.pivot(index='AGE_GROUP', columns='INCOME_GROUP', 
                                              values='POLICY_STATUS')

            fig = px.imshow(heatmap_pivot, text_auto='.1f', aspect='auto',
                           color_continuous_scale='RdYlGn_r',
                           title='Repudiation Rate: Age Group vs Income Group (%)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="warning-box">🔥 Darker red areas indicate higher bias. '
                       'Notice the concentration of high repudiation in specific age-income combinations.</div>',
                       unsafe_allow_html=True)

    # ============================================
    # PREDICTIVE MODELING
    # ============================================
    elif page == "🤖 Predictive Modeling":
        st.markdown('<div class="main-header">🤖 Supervised Learning: Classification Models</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        This section applies advanced machine learning algorithms (KNN, Decision Tree, Random Forest, 
        Gradient Boosting) to predict claim repudiation and identify key predictive features.
        """)

        model_select = st.selectbox("Select Model to Explore:", 
                                     ['All Models Overview', 'KNN', 'Decision Tree', 
                                      'Random Forest', 'Gradient Boosting'])

        if model_select == 'All Models Overview':
            st.markdown('<div class="sub-header">📊 Model Performance Summary</div>',
                       unsafe_allow_html=True)

            # Create summary table
            summary_data = []
            for name, res in results.items():
                summary_data.append({
                    'Model': name,
                    'Train Accuracy': f"{res['train_acc']:.4f}",
                    'Test Accuracy': f"{res['test_acc']:.4f}",
                    'Precision': f"{res['precision']:.4f}",
                    'Recall': f"{res['recall']:.4f}",
                    'F1-Score': f"{res['f1']:.4f}",
                    'ROC-AUC': f"{res['roc_auc']:.4f}"
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Training vs Testing Accuracy
            fig = go.Figure()
            models_list = list(results.keys())
            fig.add_trace(go.Bar(name='Training Accuracy', x=models_list,
                               y=[results[m]['train_acc'] for m in models_list],
                               marker_color='steelblue'))
            fig.add_trace(go.Bar(name='Testing Accuracy', x=models_list,
                               y=[results[m]['test_acc'] for m in models_list],
                               marker_color='coral'))
            fig.update_layout(barmode='group', title='Training vs Testing Accuracy',
                            yaxis_title='Accuracy', xaxis_title='Model')
            st.plotly_chart(fig, use_container_width=True)

            # Precision, Recall, F1
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Precision', x=models_list,
                               y=[results[m]['precision'] for m in models_list],
                               marker_color='green'))
            fig.add_trace(go.Bar(name='Recall', x=models_list,
                               y=[results[m]['recall'] for m in models_list],
                               marker_color='orange'))
            fig.add_trace(go.Bar(name='F1-Score', x=models_list,
                               y=[results[m]['f1'] for m in models_list],
                               marker_color='purple'))
            fig.update_layout(barmode='group', title='Precision, Recall & F1-Score Comparison',
                            yaxis_title='Score', xaxis_title='Model')
            st.plotly_chart(fig, use_container_width=True)

        else:
            res = results[model_select]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test Accuracy", f"{res['test_acc']:.4f}")
            with col2:
                st.metric("Precision", f"{res['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{res['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{res['f1']:.4f}")

            st.markdown('<div class="sub-header">Confusion Matrix</div>', unsafe_allow_html=True)

            # Confusion Matrix
            cm = res['cm']
            fig = px.imshow(cm, text_auto=True, aspect='equal',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Approved', 'Repudiated'],
                           y=['Approved', 'Repudiated'],
                           color_continuous_scale='Blues',
                           title=f'Confusion Matrix - {model_select}')
            st.plotly_chart(fig, use_container_width=True)

            # Classification Report
            st.markdown('<div class="sub-header">Classification Report</div>', unsafe_allow_html=True)
            report = classification_report(res['y_test'], res['y_test_pred'], 
                                          target_names=['Approved', 'Repudiated'],
                                          output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)

            # ROC Curve
            st.markdown('<div class="sub-header">ROC Curve</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'],
                                   mode='lines', name=f'{model_select} (AUC={res["roc_auc"]:.3f})',
                                   line=dict(width=2)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                   name='Random Classifier', line=dict(dash='dash', color='gray')))
            fig.update_layout(title=f'ROC Curve - {model_select}',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # MODEL COMPARISON
    # ============================================
    elif page == "📈 Model Comparison":
        st.markdown('<div class="main-header">📈 Comprehensive Model Comparison</div>',
                    unsafe_allow_html=True)

        # ROC Curves Comparison
        st.markdown('<div class="sub-header">ROC Curves Comparison</div>', unsafe_allow_html=True)
        fig = go.Figure()
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        for i, (name, res) in enumerate(results.items()):
            fig.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'], mode='lines',
                                   name=f'{name} (AUC={res["roc_auc"]:.3f})',
                                   line=dict(color=colors[i], width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                               name='Random Classifier', line=dict(dash='dash', color='gray')))
        fig.update_layout(title='ROC Curves - All Models',
                        xaxis_title='False Positive Rate',
                        yaxis_title='True Positive Rate',
                        height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrices Side by Side
        st.markdown('<div class="sub-header">Confusion Matrices - All Models</div>', 
                   unsafe_allow_html=True)

        cols = st.columns(2)
        for idx, (name, res) in enumerate(results.items()):
            with cols[idx % 2]:
                fig = px.imshow(res['cm'], text_auto=True, aspect='equal',
                               labels=dict(x="Predicted", y="Actual"),
                               x=['Approved', 'Repudiated'],
                               y=['Approved', 'Repudiated'],
                               color_continuous_scale='Blues',
                               title=f'{name}')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Feature Importance
        st.markdown('<div class="sub-header">Feature Importance Analysis</div>', 
                   unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Random Forest Feature Importance**")
            fig = px.bar(rf_imp, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Gradient Boosting Feature Importance**")
            fig = px.bar(gb_imp, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)

    # ============================================
    # FINDINGS & RECOMMENDATIONS
    # ============================================
    elif page == "📋 Findings & Recommendations":
        st.markdown('<div class="main-header">📋 Key Findings & Recommendations</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        Based on the comprehensive analysis of 1,790 insurance claims, the following critical 
        findings have been identified regarding potential bias in claim settlements:
        """)

        # Critical Findings
        st.markdown('<div class="sub-header">🚨 Critical Bias Findings</div>', unsafe_allow_html=True)

        findings = [
            {
                "title": "1. Payment Mode Discrimination",
                "severity": "HIGH",
                "detail": "Quarterly payment mode claimants face 55.0% repudiation rate vs only 10.1% for Single payment mode. "
                         "Half-Yearly mode also shows high repudiation at 48.0%.",
                "recommendation": "Review underwriting and claim processing protocols for non-annual payment modes. "
                                "Implement standardized evaluation criteria independent of payment frequency."
            },
            {
                "title": "2. Income-Based Discrimination",
                "severity": "HIGH",
                "detail": "Low income group (₹1-1L) has 51.9% repudiation rate compared to 22.4% for very high income (₹5L+). "
                         "Zero income group shows 31.5% repudiation.",
                "recommendation": "Investigate whether income verification processes create barriers for low-income claimants. "
                                "Consider socioeconomic factors in claim evaluation training."
            },
            {
                "title": "3. Zone/Team Disparities",
                "severity": "HIGH",
                "detail": "PENINSULAR (76.9%), JKB JAMMU (56.5%), and South (54.7%) show extremely high repudiation rates. "
                         "In contrast, TEAM HIMALAYAN (7.7%), RAJASTAN (7.4%), and JKB CREDITOR (3.4%) have very low rates.",
                "recommendation": "Conduct audit of zone-specific claim processing practices. "
                                "Standardize training and evaluation criteria across all zones."
            },
            {
                "title": "4. Early vs Non-Early Claim Bias",
                "severity": "MEDIUM",
                "detail": "Non-Early claims are repudiated at 37.2% vs 23.0% for Early claims. "
                         "This 14.2 percentage point gap suggests systematic preference.",
                "recommendation": "Review early claim incentive structures that may bias decision-making. "
                                "Ensure independent evaluation of claim validity regardless of timing."
            },
            {
                "title": "5. Age Group Anomalies",
                "severity": "MEDIUM",
                "detail": "The 30-40 age group shows highest repudiation (36.7%), while very young (<30: 19.6%) "
                         "and very old (70+: 22.7%) have lower rates. This U-shaped pattern is unusual.",
                "recommendation": "Analyze claim types and documentation requirements for the 30-40 age group. "
                                "Consider if this group faces unique verification challenges."
            },
            {
                "title": "6. Medical vs Non-Medical Documentation Gap",
                "severity": "MEDIUM",
                "detail": "Medical cases have lower repudiation (18.9%) vs Non-Medical (33.6%). "
                         "This 14.7 point gap may indicate better documentation standards for medical cases.",
                "recommendation": "Improve documentation support for non-medical claimants. "
                                "Provide clearer guidelines on required evidence."
            }
        ]

        for finding in findings:
            color = "#dc3545" if finding["severity"] == "HIGH" else "#ffc107"
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 15px; margin: 10px 0; 
                        background-color: #f8f9fa; border-radius: 5px;">
                <h4 style="color: {color}; margin: 0;">{finding["title"]} 
                <span style="background: {color}; color: white; padding: 2px 8px; 
                             border-radius: 10px; font-size: 0.8em;">{finding["severity"]}</span></h4>
                <p><b>Detail:</b> {finding["detail"]}</p>
                <p><b>💡 Recommendation:</b> {finding["recommendation"]}</p>
            </div>
            """, unsafe_allow_html=True)

        # Model Performance Insights
        st.markdown('<div class="sub-header">🤖 Machine Learning Model Insights</div>',
                   unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        <b>Best Performing Model: Random Forest</b><br>
        - Test Accuracy: 73.4%<br>
        - ROC-AUC: 0.744<br>
        - Most important features: SUM_ASSURED (30.0%), PI_AGE (16.1%), ZONE_CATEGORY (10.6%)<br>
        - The model confirms that Sum Assured, Age, and Zone are the strongest predictors of repudiation
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Model Stability Analysis:</b><br>
        - Random Forest shows best generalization (Train: 88.2%, Test: 73.4%)<br>
        - Gradient Boosting shows signs of overfitting (Train: 95.7%, Test: 69.2%)<br>
        - KNN performs poorly, indicating non-linear relationships in data<br>
        - Decision Tree provides good interpretability with reasonable performance
        </div>
        """, unsafe_allow_html=True)

        # Action Plan
        st.markdown('<div class="sub-header">📋 Recommended Action Plan</div>', 
                   unsafe_allow_html=True)

        st.markdown("""
        | Priority | Action Item | Timeline | Responsible |
        |----------|-------------|----------|-------------|
        | 🔴 Critical | Audit quarterly payment mode processing | 2 weeks | Claims Manager |
        | 🔴 Critical | Standardize zone evaluation criteria | 1 month | Operations Head |
        | 🟠 High | Implement income-blind claim review | 3 weeks | Compliance Team |
        | 🟠 High | Retrain staff on unbiased evaluation | 1 month | HR & Training |
        | 🟡 Medium | Enhance documentation support | 6 weeks | Customer Service |
        | 🟡 Medium | Deploy ML model for bias detection | 2 months | IT & Analytics |
        | 🟢 Low | Monthly bias monitoring dashboard | 3 months | Analytics Team |
        """)

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
        <i>Report generated by Insurance Claims Bias Analysis Dashboard | 
        For internal use only | Confidential</i>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load data. Please check that 'Insurance.csv' is in the repository root.")
    st.info("""
    **To deploy this app:**
    1. Create a GitHub repository
    2. Upload `app.py` and `Insurance.csv` to the repository
    3. Go to [share.streamlit.io](https://share.streamlit.io)
    4. Connect your GitHub repository
    5. Deploy!
    """)
