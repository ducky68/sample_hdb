import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page and styling
st.set_page_config(layout="wide", page_title="HDB Analytics Dashboard")

# Custom CSS for larger fonts and better styling
st.markdown("""
<style>
    .main > div {
        font-size: 18px;
    }
    .stMetric {
        background-color: #f0f8ff;
        border: 1px solid #e6f3ff;
        border-radius: 10px;
        padding: 10px;
    }
    .stMetric > div {
        font-size: 20px;
    }
    .stTab {
        font-size: 20px;
        font-weight: bold;
    }
    h1 {
        font-size: 3rem !important;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        font-size: 2.5rem !important;
        color: #1e40af;
        margin-bottom: 1.5rem;
    }
    h3 {
        font-size: 2rem !important;
        color: #2563eb;
        margin-bottom: 1rem;
    }
    .stMarkdown p {
        font-size: 18px !important;
    }
    .stCaption {
        font-size: 16px !important;
        font-weight: 500;
    }
    .sidebar .sidebar-content {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 HDB Resale Price Analytics Dashboard")

# Load data with caching and error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('datasets/train.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'datasets/train.csv' exists.")
        return None

df = load_data()

if df is None:
    st.stop()

# Data validation
if df.empty:
    st.error("Dataset is empty.")
    st.stop()

# Set color palette and matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Purple
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'info': '#4ECDC4',         # Teal
    'warning': '#FFE66D',      # Yellow
    'dark': '#2C3E50',         # Dark blue
    'light': '#ECF0F1'         # Light gray
}

# Color palette for charts
chart_colors = [colors['primary'], colors['secondary'], colors['accent'], 
                colors['info'], colors['success'], colors['warning']]

# Set default matplotlib parameters
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Helper: Numeric and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Sidebar Filters
st.sidebar.header("🔧 Filter Listings")

# Location filter
if 'town' in df.columns:
    selected_town = st.sidebar.multiselect(
        "🏙️ Select Town(s)",
        options=sorted(df['town'].unique()),
        default=sorted(df['town'].unique())[:3]
    )
else:
    selected_town = []
    st.sidebar.warning("Town column not found")

# Flat type filter
if 'flat_type' in df.columns:
    selected_flat_type = st.sidebar.multiselect(
        "🏠 Select Flat Type(s)",
        options=sorted(df['flat_type'].unique()),
        default=sorted(df['flat_type'].unique())
    )
else:
    selected_flat_type = []
    st.sidebar.warning("Flat type column not found")

# Price range filter
if 'resale_price' in df.columns and not df['resale_price'].empty:
    min_price, max_price = int(df['resale_price'].min()), int(df['resale_price'].max())
    price_range = st.sidebar.slider(
        "💰 Select Resale Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=10000
    )
else:
    price_range = (0, 1000000)
    st.sidebar.warning("Resale price column not found or empty")

# Floor area filter
if 'floor_area_sqm' in df.columns and not df['floor_area_sqm'].empty:
    min_area, max_area = int(df['floor_area_sqm'].min()), int(df['floor_area_sqm'].max())
    area_range = st.sidebar.slider(
        "📐 Select Floor Area Range (sqm)",
        min_value=min_area,
        max_value=max_area,
        value=(min_area, max_area),
        step=1
    )
else:
    area_range = None
    st.sidebar.warning("Floor area column not found or empty")

# Filter dataframe with error handling
try:
    filtered_df = df.copy()
    
    if selected_town and 'town' in df.columns:
        filtered_df = filtered_df[filtered_df['town'].isin(selected_town)]
    
    if selected_flat_type and 'flat_type' in df.columns:
        filtered_df = filtered_df[filtered_df['flat_type'].isin(selected_flat_type)]
    
    if 'resale_price' in df.columns:
        filtered_df = filtered_df[filtered_df['resale_price'].between(*price_range)]
    
    if area_range and 'floor_area_sqm' in df.columns:
        filtered_df = filtered_df[filtered_df['floor_area_sqm'].between(*area_range)]
        
except Exception as e:
    st.error(f"Error filtering data: {e}")
    filtered_df = df.copy()

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data matches your current filters. Please adjust your selections.")
    filtered_df = df.copy()

# Tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💡 Actionable Insights", "🔍 Unusual Insights", "💰 Profit Insights"])

with tab1:
    st.header("Overview of Filtered Listings")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Listings", len(filtered_df))
    with col2:
        if 'resale_price' in filtered_df.columns and not filtered_df.empty:
            st.metric("Median Price", f"${filtered_df['resale_price'].median():,.0f}")
        else:
            st.metric("Median Price", "N/A")
    with col3:
        if 'floor_area_sqm' in filtered_df.columns and not filtered_df.empty:
            st.metric("Median Floor Area", f"{filtered_df['floor_area_sqm'].median():.0f} sqm")
        else:
            st.metric("Median Floor Area", "N/A")
    with col4:
        if 'resale_price' in filtered_df.columns and 'floor_area_sqm' in filtered_df.columns and not filtered_df.empty:
            price_per_sqm = (filtered_df['resale_price'] / filtered_df['floor_area_sqm']).median()
            st.metric("Median Price/sqm", f"${price_per_sqm:.0f}")
        else:
            st.metric("Median Price/sqm", "N/A")
    
    # Display sample data
    st.subheader("Sample Listings")
    if not filtered_df.empty:
        display_cols = ['town', 'flat_type', 'resale_price', 'floor_area_sqm', 'block', 'street_name']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        st.dataframe(filtered_df[available_cols].head(100), use_container_width=True)
    else:
        st.info("No data to display with current filters.")

with tab2:
    st.header("💡 5 Actionable Insights (Filtered Data)")
    
    if filtered_df.empty:
        st.warning("No data available for analysis with current filters.")
    else:
        # 1. Median Resale Price by Flat Type
        st.subheader("1. Median Resale Price by Flat Type")
        if 'flat_type' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            flat_type_price = filtered_df.groupby('flat_type')['resale_price'].median().sort_values()
            if not flat_type_price.empty:
                # Create colorful bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(flat_type_price.index, flat_type_price.values, 
                             color=chart_colors[:len(flat_type_price)], alpha=0.8)
                ax.set_title('Median Resale Price by Flat Type', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.set_xlabel('Flat Type', fontsize=14, fontweight='bold')
                ax.set_ylabel('Median Resale Price ($)', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("💡 Larger flat types (e.g., Executive, 5 ROOM) command higher median resale prices. Consider targeting these for higher returns.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 2. Top 10 Towns by Median Resale Price
        st.subheader("2. Top 10 Towns by Median Resale Price")
        if 'town' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            town_price = filtered_df.groupby('town')['resale_price'].median().sort_values(ascending=False).head(10)
            if not town_price.empty:
                # Create horizontal bar chart with gradient colors
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.barh(town_price.index, town_price.values, 
                              color=sns.color_palette("viridis", len(town_price)))
                ax.set_title('Top 10 Towns by Median Resale Price', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.set_xlabel('Median Resale Price ($)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Town', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars, town_price.values)):
                    ax.text(value + max(town_price.values) * 0.01, bar.get_y() + bar.get_height()/2,
                           f'${value:,.0f}', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("💡 Some towns have much higher median prices. Location is a key driver of value.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 3. Floor Area vs Resale Price
        st.subheader("3. Floor Area vs Resale Price")
        if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(filtered_df['floor_area_sqm'], filtered_df['resale_price'], 
                                   alpha=0.6, c=colors['primary'], s=50, edgecolors='white', linewidth=0.5)
                ax.set_xlabel('Floor Area (sqm)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Resale Price ($)', fontsize=14, fontweight='bold')
                ax.set_title('Floor Area vs Resale Price', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(filtered_df['floor_area_sqm'], filtered_df['resale_price'], 1)
                p = np.poly1d(z)
                ax.plot(filtered_df['floor_area_sqm'].sort_values(), 
                       p(filtered_df['floor_area_sqm'].sort_values()), 
                       color=colors['accent'], linestyle='--', linewidth=2, alpha=0.8)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("💡 Larger floor area generally leads to higher resale prices, but with diminishing returns at the upper end.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 4. HDB Age vs Resale Price
        st.subheader("4. HDB Age vs Resale Price")
        if 'hdb_age' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                scatter = ax.scatter(filtered_df['hdb_age'], filtered_df['resale_price'], 
                                   alpha=0.6, c=colors['secondary'], s=50, edgecolors='white', linewidth=0.5)
                ax.set_xlabel('HDB Age (years)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Resale Price ($)', fontsize=14, fontweight='bold')
                ax.set_title('HDB Age vs Resale Price', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(filtered_df['hdb_age'], filtered_df['resale_price'], 1)
                p = np.poly1d(z)
                ax.plot(filtered_df['hdb_age'].sort_values(), 
                       p(filtered_df['hdb_age'].sort_values()), 
                       color=colors['success'], linestyle='--', linewidth=2, alpha=0.8)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("💡 Older flats tend to have lower resale prices. Consider lease decay in investment decisions.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("HDB age column not found in the dataset.")

        # 5. Storey Range vs Resale Price
        st.subheader("5. Storey Range vs Resale Price")
        if 'storey_range' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            storey_price = filtered_df.groupby('storey_range')['resale_price'].median().sort_values()
            if not storey_price.empty:
                # Create colorful bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(storey_price.index, storey_price.values, 
                             color=sns.color_palette("plasma", len(storey_price)), alpha=0.8)
                ax.set_title('Storey Range vs Median Resale Price', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.set_xlabel('Storey Range', fontsize=14, fontweight='bold')
                ax.set_ylabel('Median Resale Price ($)', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("💡 Higher storey flats may command a price premium in some cases.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("Storey range column not found in the dataset.")

with tab3:
    st.header("🔍 5 Actionable Unusual Insights (Filtered Data)")
    
    if filtered_df.empty:
        st.warning("No data available for analysis with current filters.")
    else:
        # 1. Towns with Unexpectedly Low Median Prices
        st.subheader("1. Towns with Unexpectedly Low Median Prices")
        if 'town' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            low_town = filtered_df.groupby('town')['resale_price'].median().sort_values().head(5)
            if not low_town.empty:
                # Create colorful bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(low_town.index, low_town.values, 
                             color=sns.color_palette("Reds_r", len(low_town)), alpha=0.8)
                ax.set_title('Towns with Lowest Median Resale Prices', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.set_xlabel('Town', fontsize=14, fontweight='bold')
                ax.set_ylabel('Median Resale Price ($)', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("🔍 Some towns have much lower prices than average, which may present value opportunities.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 2. Flat Models with High Price Variance
        st.subheader("2. Flat Models with High Price Variance")
        if 'flat_model' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            model_var = filtered_df.groupby('flat_model')['resale_price'].std().sort_values(ascending=False).head(5)
            if not model_var.empty:
                # Create colorful bar chart
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(model_var.index, model_var.values, 
                             color=sns.color_palette("coolwarm", len(model_var)), alpha=0.8)
                ax.set_title('Flat Models with Highest Price Variance', fontsize=18, fontweight='bold', color=colors['dark'])
                ax.set_xlabel('Flat Model', fontsize=14, fontweight='bold')
                ax.set_ylabel('Price Standard Deviation ($)', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                st.caption("🔍 Certain flat models show high price variance, indicating diverse market segments.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("Flat model column not found in the dataset.")

        # 3. Top 5 Flats by Price per Square Meter
        st.subheader("3. Top 5 Flats by Price per Square Meter")
        if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                df_with_price_per_sqm = filtered_df.copy()
                df_with_price_per_sqm['price_per_sqm'] = df_with_price_per_sqm['resale_price'] / df_with_price_per_sqm['floor_area_sqm']
                
                display_cols = ['block', 'street_name', 'price_per_sqm']
                available_cols = [col for col in display_cols if col in df_with_price_per_sqm.columns]
                
                if available_cols:
                    top_ppsqm = df_with_price_per_sqm.nlargest(5, 'price_per_sqm')[available_cols]
                    st.dataframe(top_ppsqm, use_container_width=True)
                    st.caption("🔍 Some flats achieve extremely high price per sqm, possibly due to location or unique features.")
                else:
                    st.info("Required columns not available for display.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 4. Flats with Oldest Lease Commence Dates
        st.subheader("4. Flats with Oldest Lease Commence Dates")
        if 'lease_commence_date' in filtered_df.columns:
            if len(filtered_df) > 0:
                display_cols = ['block', 'street_name', 'lease_commence_date', 'resale_price']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                
                if available_cols:
                    oldest = filtered_df.nsmallest(5, 'lease_commence_date')[available_cols]
                    st.dataframe(oldest, use_container_width=True)
                    st.caption("🔍 Some very old flats are still transacting, which may indicate heritage or special demand.")
                else:
                    st.info("Required columns not available for display.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("Lease commence date column not found in the dataset.")

        # 5. Top 5 Most Expensive Flats
        st.subheader("5. Top 5 Most Expensive Flats")
        if 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                display_cols = ['block', 'street_name', 'resale_price']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                
                if available_cols:
                    top_exp = filtered_df.nlargest(5, 'resale_price')[available_cols]
                    st.dataframe(top_exp, use_container_width=True)
                    st.caption("🔍 A small number of flats sell for much higher than the rest, likely due to rare attributes.")
                else:
                    st.info("Required columns not available for display.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Resale price column not found.")

with tab4:
    st.header("💰 5 High-Profit Business Insights")
    
    if filtered_df.empty:
        st.warning("No data available for analysis with current filters.")
    else:
        # 1. Undervalued Properties by Price per sqm
        st.subheader("1. Undervalued Properties (Low Price per sqm)")
        if 'floor_area_sqm' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                df_with_price_per_sqm = filtered_df.copy()
                df_with_price_per_sqm['price_per_sqm'] = df_with_price_per_sqm['resale_price'] / df_with_price_per_sqm['floor_area_sqm']
                
                # Find bottom 10% by price per sqm
                percentile_10 = df_with_price_per_sqm['price_per_sqm'].quantile(0.1)
                undervalued = df_with_price_per_sqm[df_with_price_per_sqm['price_per_sqm'] <= percentile_10]
                
                if len(undervalued) > 0:
                    avg_by_town = undervalued.groupby('town').agg({
                        'price_per_sqm': 'mean',
                        'resale_price': 'count'
                    }).round(0)
                    avg_by_town.columns = ['Avg Price/sqm', 'Count']
                    avg_by_town = avg_by_town.sort_values('Count', ascending=False).head(10)
                    
                    st.dataframe(avg_by_town, use_container_width=True)
                    st.caption("💰 **PROFIT OPPORTUNITY**: These towns have the most undervalued properties. Target for flipping or rental yield optimization.")
                else:
                    st.info("No undervalued properties found with current filters.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 2. High-Growth Potential Areas (Young HDBs in Prime Locations)
        st.subheader("2. High-Growth Potential Areas")
        if 'hdb_age' in filtered_df.columns and 'town' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                # Find relatively new flats (< 20 years) in high-value towns
                young_flats = filtered_df[filtered_df['hdb_age'] < 20] if 'hdb_age' in filtered_df.columns else filtered_df
                
                if len(young_flats) > 0:
                    growth_potential = young_flats.groupby('town').agg({
                        'resale_price': ['mean', 'count'],
                        'hdb_age': 'mean'
                    }).round(0)
                    growth_potential.columns = ['Avg Price', 'Count', 'Avg Age']
                    growth_potential = growth_potential[growth_potential['Count'] >= 5]  # At least 5 properties
                    growth_potential = growth_potential.sort_values('Avg Price', ascending=False).head(8)
                    
                    st.dataframe(growth_potential, use_container_width=True)
                    st.caption("💰 **PROFIT OPPORTUNITY**: Young flats in these high-value towns have strong appreciation potential. Ideal for long-term investment.")
                else:
                    st.info("No young properties found with current filters.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("Required columns not found for this analysis.")

        # 3. Premium Floor Analysis (High Storey Premium)
        st.subheader("3. Premium Floor Investment Strategy")
        if 'storey_range' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                storey_analysis = filtered_df.groupby('storey_range').agg({
                    'resale_price': ['mean', 'count']
                }).round(0)
                storey_analysis.columns = ['Avg Price', 'Count']
                storey_analysis = storey_analysis[storey_analysis['Count'] >= 10]
                storey_analysis['Premium_vs_Low'] = storey_analysis['Avg Price'] - storey_analysis['Avg Price'].min()
                storey_analysis = storey_analysis.sort_values('Premium_vs_Low', ascending=False).head(8)
                
                st.dataframe(storey_analysis, use_container_width=True)
                st.caption("💰 **PROFIT OPPORTUNITY**: High-floor units command significant premiums. Target mid-high floors for best ROI balance.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.info("Storey range column not found in the dataset.")

        # 4. Optimal Flat Type for ROI
        st.subheader("4. Best Flat Types for Investment ROI")
        if 'flat_type' in filtered_df.columns and 'resale_price' in filtered_df.columns and 'floor_area_sqm' in filtered_df.columns:
            if len(filtered_df) > 0:
                roi_analysis = filtered_df.groupby('flat_type').agg({
                    'resale_price': ['mean', 'median', 'count'],
                    'floor_area_sqm': 'mean'
                }).round(0)
                roi_analysis.columns = ['Avg Price', 'Median Price', 'Count', 'Avg Area']
                roi_analysis['Price_per_sqm'] = roi_analysis['Avg Price'] / roi_analysis['Avg Area']
                roi_analysis = roi_analysis[roi_analysis['Count'] >= 20]
                roi_analysis = roi_analysis.sort_values('Price_per_sqm', ascending=True).head(6)
                
                st.dataframe(roi_analysis, use_container_width=True)
                st.caption("💰 **PROFIT OPPORTUNITY**: These flat types offer the best value per sqm. Lower entry cost with good rental/resale potential.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # 5. Market Gap Analysis (Low Supply, High Demand Areas)
        st.subheader("5. Market Gap Analysis")
        if 'town' in filtered_df.columns and 'flat_type' in filtered_df.columns and 'resale_price' in filtered_df.columns:
            if len(filtered_df) > 0:
                # Find town-flat_type combinations with low supply but high prices
                market_analysis = filtered_df.groupby(['town', 'flat_type']).agg({
                    'resale_price': ['mean', 'count']
                }).round(0)
                market_analysis.columns = ['Avg Price', 'Supply Count']
                
                # Filter for low supply (< 10) but high price areas
                gap_opportunities = market_analysis[
                    (market_analysis['Supply Count'] < 10) & 
                    (market_analysis['Avg Price'] > market_analysis['Avg Price'].median())
                ].sort_values('Avg Price', ascending=False).head(10)
                
                if len(gap_opportunities) > 0:
                    st.dataframe(gap_opportunities, use_container_width=True)
                    st.caption("💰 **PROFIT OPPORTUNITY**: Low supply + High prices = Market gaps. These combinations indicate unmet demand and pricing power.")
                else:
                    # Show alternative analysis if no gaps found
                    high_demand = market_analysis.nlargest(10, 'Avg Price')
                    st.dataframe(high_demand, use_container_width=True)
                    st.caption("💰 **PROFIT OPPORTUNITY**: Highest priced segments indicate strong demand and premium positioning potential.")
            else:
                st.info("No data available for this analysis.")
        else:
            st.warning("Required columns not found for this analysis.")

        # Summary Box
        st.markdown("---")
        st.info("""
        **🎯 KEY PROFIT STRATEGIES:**
        1. **Buy Low**: Target undervalued properties in growing areas
        2. **Hold Long**: Invest in young flats in prime locations  
        3. **Go High**: Premium floors command higher rents/resale
        4. **Size Smart**: Choose flat types with best value per sqm
        5. **Fill Gaps**: Enter markets with low supply but high demand
        """)

# Footer
st.markdown("---")
st.success("🎉 EDA with actionable insights complete! Use the sidebar filters to explore different segments of the data.")
st.info("💡 Tip: Try different combinations of filters to discover unique market opportunities.")
