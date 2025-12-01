
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import time
from datetime import datetime

sys.path.append('./src')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENHANCED STYLING
# ============================================================================

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

def apply_theme():
    if st.session_state.theme == 'dark':
        bg = "#0E1117"
        sec_bg = "#262730"
        text = "#FAFAFA"
        card = "#1E1E1E"
    else:
        bg = "#FFFFFF"
        sec_bg = "#F0F2F6"
        text = "#262730"
        card = "#FFFFFF"
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, {bg} 0%, {sec_bg} 100%);
        font-family: 'Poppins', sans-serif;
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {sec_bg} 0%, {bg} 100%);
        border-right: 2px solid #FF4B4B;
    }}
    
    h1, h2, h3 {{ color: {text}; font-family: 'Poppins', sans-serif; font-weight: 700; }}
    
    .stButton>button {{
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white; border-radius: 30px; border: none;
        padding: 0.75rem 2.5rem; font-weight: 700; font-size: 1.1rem;
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase; letter-spacing: 1px;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(255, 75, 75, 0.5);
    }}
    
    .price-card {{
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 50%, #FF8E53 100%);
        border-radius: 25px; padding: 3rem; text-align: center;
        box-shadow: 0 15px 40px rgba(255, 75, 75, 0.4);
        margin: 2rem 0; position: relative; overflow: hidden;
    }}
    
    .price-card::before {{
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1) rotate(0deg); }}
        50% {{ transform: scale(1.1) rotate(180deg); }}
    }}
    
    .gradient-text {{
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 50%, #FF8E53 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-weight: 900; font-size: 2.5rem;
        animation: gradient-shift 3s ease infinite;
    }}
    
    @keyframes gradient-shift {{
        0%, 100% {{ filter: hue-rotate(0deg); }}
        50% {{ filter: hue-rotate(20deg); }}
    }}
    
    .metric-card {{
        background: {card}; border-radius: 15px; padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.2);
    }}
    
    .feature-icon {{
        font-size: 3rem; margin-bottom: 1rem;
        animation: bounce 2s ease-in-out infinite;
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-20px); }}
    }}
    
    .info-card {{
        background: linear-gradient(135deg, {card} 0%, {sec_bg} 100%);
        border-radius: 20px; padding: 2rem; margin: 1rem 0;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }}
    
    .stat-number {{
        font-size: 3rem; font-weight: 900;
        background: linear-gradient(135deg, #FF4B4B, #FF8E53);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    
    .upload-box {{
        border: 3px dashed #FF4B4B; border-radius: 20px;
        padding: 3rem; text-align: center; background: {card};
        transition: all 0.3s ease;
    }}
    
    .upload-box:hover {{
        border-color: #FF6B6B; background: {sec_bg};
        transform: scale(1.02);
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 2rem; font-weight: 700; color: #FF4B4B;
    }}
    
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ============================================================================
# LOAD RESOURCES
# ============================================================================

@st.cache_resource
def load_all_resources():
    try:
        with open('./models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        from preprocessing import HousePricePreprocessor
        preprocessor = HousePricePreprocessor(artifacts_path='./data/artifacts/')
        preprocessor.load_artifacts()
        return model, metadata, preprocessor
    except Exception as e:
        st.error(f"❌ Error loading: {e}")
        return None, None, None

model, metadata, preprocessor = load_all_resources()

@st.cache_data
def load_samples():
    try:
        return pd.read_csv('./data/sample_houses.csv')
    except:
        return None

samples = load_samples()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_price(price):
    return f"${price:,.0f}"

def create_raw_input_dataframe(user_inputs):
    """Create complete raw dataframe from user inputs"""
    raw_data = {
        'Id': [9999],
        'MSSubClass': [20], 'MSZoning': ['RL'],
        'LotFrontage': [user_inputs.get('lot_area', 9000) ** 0.5],
        'LotArea': [user_inputs['lot_area']],
        'Street': ['Pave'], 'Alley': [np.nan], 'LotShape': ['Reg'],
        'LandContour': ['Lvl'], 'Utilities': ['AllPub'], 'LotConfig': ['Inside'],
        'LandSlope': ['Gtl'], 'Neighborhood': ['NAmes'],
        'Condition1': ['Norm'], 'Condition2': ['Norm'],
        'BldgType': ['1Fam'],
        'HouseStyle': ['2Story' if user_inputs['second_flr_sf'] > 0 else '1Story'],
        'OverallQual': [user_inputs['overall_qual']],
        'OverallCond': [user_inputs['overall_cond']],
        'YearBuilt': [user_inputs['year_built']],
        'YearRemodAdd': [user_inputs['year_remod']],
        'RoofStyle': ['Gable'], 'RoofMatl': ['CompShg'],
        'Exterior1st': ['VinylSd'], 'Exterior2nd': ['VinylSd'],
        'MasVnrType': ['None'], 'MasVnrArea': [0],
        'ExterQual': ['TA'], 'ExterCond': ['TA'], 'Foundation': ['PConc'],
        'BsmtQual': ['TA' if user_inputs['total_bsmt_sf'] > 0 else np.nan],
        'BsmtCond': ['TA' if user_inputs['total_bsmt_sf'] > 0 else np.nan],
        'BsmtExposure': ['No' if user_inputs['total_bsmt_sf'] > 0 else np.nan],
        'BsmtFinType1': ['Unf' if user_inputs['total_bsmt_sf'] > 0 else np.nan],
        'BsmtFinSF1': [user_inputs['total_bsmt_sf'] * 0.5 if user_inputs['total_bsmt_sf'] > 0 else 0],
        'BsmtFinType2': ['Unf' if user_inputs['total_bsmt_sf'] > 0 else np.nan],
        'BsmtFinSF2': [0],
        'BsmtUnfSF': [user_inputs['total_bsmt_sf'] * 0.5 if user_inputs['total_bsmt_sf'] > 0 else 0],
        'TotalBsmtSF': [user_inputs['total_bsmt_sf']],
        'Heating': ['GasA'], 'HeatingQC': ['Ex'], 'CentralAir': ['Y'],
        'Electrical': ['SBrkr'],
        '1stFlrSF': [user_inputs['first_flr_sf']],
        '2ndFlrSF': [user_inputs['second_flr_sf']],
        'LowQualFinSF': [0], 'GrLivArea': [user_inputs['gr_liv_area']],
        'BsmtFullBath': [1 if user_inputs['total_bsmt_sf'] > 0 else 0],
        'BsmtHalfBath': [0],
        'FullBath': [user_inputs['full_bath']], 'HalfBath': [0],
        'BedroomAbvGr': [user_inputs['bedroom']],
        'KitchenAbvGr': [user_inputs['kitchen']],
        'KitchenQual': ['TA'], 'TotRmsAbvGrd': [user_inputs['tot_rooms']],
        'Functional': ['Typ'], 'Fireplaces': [0], 'FireplaceQu': [np.nan],
        'GarageType': ['Attchd' if user_inputs['garage_area'] > 0 else np.nan],
        'GarageYrBlt': [user_inputs['year_built'] if user_inputs['garage_area'] > 0 else np.nan],
        'GarageFinish': ['Unf' if user_inputs['garage_area'] > 0 else np.nan],
        'GarageCars': [user_inputs['garage_cars']],
        'GarageArea': [user_inputs['garage_area']],
        'GarageQual': ['TA' if user_inputs['garage_area'] > 0 else np.nan],
        'GarageCond': ['TA' if user_inputs['garage_area'] > 0 else np.nan],
        'PavedDrive': ['Y'], 'WoodDeckSF': [0], 'OpenPorchSF': [0],
        'EnclosedPorch': [0], '3SsnPorch': [0], 'ScreenPorch': [0],
        'PoolArea': [0], 'PoolQC': [np.nan], 'Fence': [np.nan],
        'MiscFeature': [np.nan], 'MiscVal': [0],
        'MoSold': [6], 'YrSold': [user_inputs['yr_sold']],
        'SaleType': ['WD'], 'SaleCondition': ['Normal']
    }
    return pd.DataFrame(raw_data)

def process_and_predict(user_inputs):
    """Process inputs and make prediction"""
    raw_df = create_raw_input_dataframe(user_inputs)
    processed_df, _ = preprocessor.preprocess_pipeline(raw_df, is_train=False, apply_pca_flag=False)
    features = processed_df.drop(['Id'], axis=1, errors='ignore')
    
    for col in metadata['features']:
        if col not in features.columns:
            features[col] = 0
    
    extra = set(features.columns) - set(metadata['features'])
    if extra:
        features = features.drop(columns=list(extra))
    
    features = features[metadata['features']]
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    prediction_log = model.predict(features)[0]
    prediction = np.expm1(prediction_log)
    
    return prediction, prediction_log

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    # Theme toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🎨 Theme")
    with col2:
        if st.button("🌓", key="theme_btn"):
            toggle_theme()
            st.rerun()
    
    st.markdown("---")
    
    # Animated logo
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <div class='feature-icon'>🏠</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='gradient-text' style='text-align: center; font-size: 1.8rem;'>House Price Predictor</h1>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model stats
    if metadata:
        st.markdown("### 📊 Model Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🤖 Model", metadata['model_name'].split()[0])
        with col2:
            st.metric("🎯 Features", metadata['n_features'])
        
        st.metric("✨ Accuracy", f"{metadata['val_r2']:.1%}")
        st.progress(metadata['val_r2'])
        
        st.metric("📉 Avg Error", f"${metadata['val_rmse']:,.0f}")
        
        st.markdown(f"""
        <div class='info-card' style='margin-top: 1rem; padding: 1rem;'>
            <p style='margin: 0; font-size: 0.9rem;'>
                <b>🎓 Trained on:</b> 1,460 houses<br>
                <b>📅 Last updated:</b> {datetime.now().strftime('%b %Y')}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "",
        ["🏠 Single Prediction", "📦 Batch Predictions", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8rem;'>
        <p><b>Built with ❤️</b></p>
        <p>Streamlit + ML</p>
        <p>© 2024</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 1: SINGLE PREDICTION
# ============================================================================

if page == "🏠 Single Prediction":
    st.markdown("<h1 style='text-align: center;'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'>Enter house details for instant AI-powered estimate</p>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample loader
    if samples is not None:
        with st.expander("🎯 Quick Start - Load Sample House", expanded=False):
            sample_names = samples['Name'].tolist() if 'Name' in samples.columns else [f"Sample {i+1}" for i in range(len(samples))]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox("Choose a sample:", sample_names, label_visibility="collapsed")
            with col2:
                if st.button("📥 Load", use_container_width=True):
                    st.session_state.sample_loaded = True
                    st.session_state.sample_data = samples.iloc[sample_names.index(selected)]
                    st.success(f"✅ {selected}")
                    time.sleep(0.5)
                    st.rerun()
    
    # Input form
    with st.form("predict_form"):
        sample_data = st.session_state.get('sample_data', None) if st.session_state.get('sample_loaded', False) else None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📐 Size & Structure")
            gr_liv_area = st.number_input("🏠 Living Area (sq ft)", 300, 6000, 
                int(sample_data['GrLivArea']) if sample_data is not None and 'GrLivArea' in sample_data else 1500, 100)
            total_bsmt_sf = st.number_input("🔽 Basement (sq ft)", 0, 3000,
                int(sample_data['TotalBsmtSF']) if sample_data is not None and 'TotalBsmtSF' in sample_data else 1000, 100)
            first_flr_sf = st.number_input("1️⃣ 1st Floor (sq ft)", 300, 4000,
                int(sample_data['1stFlrSF']) if sample_data is not None and '1stFlrSF' in sample_data else 1000, 100)
            second_flr_sf = st.number_input("2️⃣ 2nd Floor (sq ft)", 0, 2000,
                int(sample_data['2ndFlrSF']) if sample_data is not None and '2ndFlrSF' in sample_data else 0, 100)
            lot_area = st.number_input("🌳 Lot Area (sq ft)", 1000, 50000,
                int(sample_data['LotArea']) if sample_data is not None and 'LotArea' in sample_data else 9000, 500)
        
        with col2:
            st.markdown("### ⭐ Quality & Condition")
            overall_qual = st.slider("🏆 Overall Quality (1-10)", 1, 10,
                int(sample_data['OverallQual']) if sample_data is not None and 'OverallQual' in sample_data else 5)
            overall_cond = st.slider("🔧 Overall Condition (1-10)", 1, 10,
                int(sample_data['OverallCond']) if sample_data is not None and 'OverallCond' in sample_data else 5)
            year_built = st.number_input("📅 Year Built", 1870, 2025,
                int(sample_data['YearBuilt']) if sample_data is not None and 'YearBuilt' in sample_data else 2000, 1)
            year_remod = st.number_input("🔨 Year Remodeled", 1870, 2025,
                int(sample_data['YearRemodAdd']) if sample_data is not None and 'YearRemodAdd' in sample_data else 2000, 1)
            yr_sold = st.number_input("💰 Year to Sell", 2006, 2030, 2024, 1)
        
        with col3:
            st.markdown("### 🚗 Rooms & Amenities")
            full_bath = st.number_input("🛁 Full Bathrooms", 0, 5,
                int(sample_data['FullBath']) if sample_data is not None and 'FullBath' in sample_data else 2, 1)
            bedroom = st.number_input("🛏️ Bedrooms", 0, 10,
                int(sample_data['BedroomAbvGr']) if sample_data is not None and 'BedroomAbvGr' in sample_data else 3, 1)
            kitchen = st.number_input("🍳 Kitchens", 0, 3,
                int(sample_data['KitchenAbvGr']) if sample_data is not None and 'KitchenAbvGr' in sample_data else 1, 1)
            tot_rooms = st.number_input("🚪 Total Rooms", 0, 15,
                int(sample_data['TotRmsAbvGrd']) if sample_data is not None and 'TotRmsAbvGrd' in sample_data else 7, 1)
            garage_cars = st.number_input("🚗 Garage Capacity", 0, 5,
                int(sample_data['GarageCars']) if sample_data is not None and 'GarageCars' in sample_data else 2, 1)
            garage_area = st.number_input("🏪 Garage Area (sq ft)", 0, 1500,
                int(sample_data['GarageArea']) if sample_data is not None and 'GarageArea' in sample_data else 500, 50)
        
        submitted = st.form_submit_button("🔮 PREDICT PRICE NOW", use_container_width=True)
        
        if submitted:
            with st.spinner('🔮 AI is analyzing your house...'):
                time.sleep(1.5)
            
            try:
                user_inputs = {
                    'gr_liv_area': gr_liv_area, 'total_bsmt_sf': total_bsmt_sf,
                    'first_flr_sf': first_flr_sf, 'second_flr_sf': second_flr_sf,
                    'lot_area': lot_area, 'overall_qual': overall_qual,
                    'overall_cond': overall_cond, 'year_built': year_built,
                    'year_remod': year_remod, 'yr_sold': yr_sold,
                    'full_bath': full_bath, 'bedroom': bedroom,
                    'kitchen': kitchen, 'tot_rooms': tot_rooms,
                    'garage_cars': garage_cars, 'garage_area': garage_area
                }
                
                prediction, prediction_log = process_and_predict(user_inputs)
                
                if prediction < 30000 or prediction > 800000:
                    st.warning(f"⚠️ Price ${prediction:,.0f} is outside typical range")
                
                st.balloons()
                st.success("✅ Prediction Complete!")
                
                # Main price card
                st.markdown(f"""
                <div class="price-card">
                    <h2 style="color: white; margin: 0; font-weight: 600;">Estimated Price</h2>
                    <h1 style="color: white; font-size: 4rem; margin: 1.5rem 0; font-weight: 900;">
                        {format_price(prediction)}
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                        🎯 Confidence: {metadata['val_r2']:.0%} | 📊 Based on {metadata['n_features']} features
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💎 Price per Sq Ft", f"${prediction/gr_liv_area:.2f}",
                             help="Price divided by living area")
                with col2:
                    uncertainty = metadata['val_rmse']
                    st.metric("📊 Confidence Range", f"±${uncertainty:,.0f}",
                             help=f"${prediction-uncertainty:,.0f} to ${prediction+uncertainty:,.0f}")
                with col3:
                    house_age = yr_sold - year_built
                    st.metric("🏗️ House Age", f"{house_age} years")
                with col4:
                    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
                    st.metric("📏 Total Area", f"{total_sf:,} sq ft")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Estimated Value ($)", 'font': {'size': 24, 'color': '#FF4B4B'}},
                    delta={'reference': 180000, 'increasing': {'color': "#00CC96"}},
                    gauge={
                        'axis': {'range': [None, min(prediction * 1.5, 1000000)], 'tickcolor': "#FF4B4B"},
                        'bar': {'color': "#FF4B4B", 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#FF4B4B",
                        'steps': [
                            {'range': [0, prediction * 0.7], 'color': 'rgba(255, 75, 75, 0.1)'},
                            {'range': [prediction * 0.7, prediction * 1.3], 'color': 'rgba(255, 75, 75, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "#00CC96", 'width': 6},
                            'thickness': 0.8,
                            'value': prediction
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FAFAFA" if st.session_state.theme == 'dark' else "#262730", 'size': 16},
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                st.markdown("### 💡 Price Insights")
                
                insights = []
                price_per_sqft = prediction / gr_liv_area
                
                if price_per_sqft > 150:
                    insights.append("🔥 **Premium Location** - High price per square foot suggests desirable area")
                elif price_per_sqft < 80:
                    insights.append("💰 **Value Opportunity** - Lower price per sq ft may indicate room for appreciation")
                
                if house_age < 5:
                    insights.append("✨ **New Construction** - Modern features and warranties")
                elif house_age > 50:
                    insights.append("🏛️ **Historic Charm** - Established neighborhood character")
                
                if overall_qual >= 8:
                    insights.append("⭐ **Premium Quality** - Excellent materials and finishes")
                
                if garage_cars >= 3:
                    insights.append("🚗 **Large Garage** - Premium feature for car enthusiasts")
                
                if total_sf > 3000:
                    insights.append("🏰 **Spacious Living** - Plenty of room for growing families")
                
                for insight in insights:
                    st.markdown(f"""
                    <div class='info-card' style='padding: 1rem;'>
                        {insight}
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PAGE 2: BATCH PREDICTIONS
# ============================================================================

elif page == "📦 Batch Predictions":
    st.markdown("<h1 style='text-align: center;'>📦 Batch House Price Predictions</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #888;'>Upload CSV file to predict multiple houses at once</p>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📋 CSV Format Instructions", expanded=False):
        st.markdown("""
        Your CSV file should contain these columns (minimum):
        - `GrLivArea` - Living area in sq ft
        - `TotalBsmtSF` - Basement area
        - `1stFlrSF` - First floor area
        - `2ndFlrSF` - Second floor area
        - `LotArea` - Lot size
        - `OverallQual` - Quality (1-10)
        - `OverallCond` - Condition (1-10)
        - `YearBuilt`, `YearRemodAdd`, `YrSold`
        - `FullBath`, `BedroomAbvGr`, `KitchenAbvGr`, `TotRmsAbvGrd`
        - `GarageCars`, `GarageArea`
        
        Missing columns will use default values automatically!
        """)
        
        st.download_button(
            "📥 Download Sample CSV Template",
            data=samples.to_csv(index=False) if samples is not None else "GrLivArea,TotalBsmtSF,1stFlrSF,2ndFlrSF,LotArea,OverallQual,OverallCond,YearBuilt,YearRemodAdd,YrSold,FullBath,BedroomAbvGr,KitchenAbvGr,TotRmsAbvGrd,GarageCars,GarageArea\n1500,1000,1000,0,9000,5,5,2000,2000,2024,2,3,1,7,2,500",
            file_name="house_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # File uploader
    st.markdown("""
    <div class='upload-box'>
        <p style='font-size: 3rem; margin: 0;'>📤</p>
        <p style='font-size: 1.5rem; font-weight: 600; color: #FF4B4B;'>Drop your CSV file here</p>
        <p style='color: #888;'>or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['csv'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {batch_df.shape[0]} houses, {batch_df.shape[1]} columns")
            
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 PREDICT ALL PRICES", use_container_width=True, type="primary"):
                with st.spinner(f'🔮 Processing {len(batch_df)} houses...'):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in batch_df.iterrows():
                        user_inputs = {
                            'gr_liv_area': row.get('GrLivArea', 1500),
                            'total_bsmt_sf': row.get('TotalBsmtSF', 1000),
                            'first_flr_sf': row.get('1stFlrSF', 1000),
                            'second_flr_sf': row.get('2ndFlrSF', 0),
                            'lot_area': row.get('LotArea', 9000),
                            'overall_qual': row.get('OverallQual', 5),
                            'overall_cond': row.get('OverallCond', 5),
                            'year_built': row.get('YearBuilt', 2000),
                            'year_remod': row.get('YearRemodAdd', 2000),
                            'yr_sold': row.get('YrSold', 2024),
                            'full_bath': row.get('FullBath', 2),
                            'bedroom': row.get('BedroomAbvGr', 3),
                            'kitchen': row.get('KitchenAbvGr', 1),
                            'tot_rooms': row.get('TotRmsAbvGrd', 7),
                            'garage_cars': row.get('GarageCars', 2),
                            'garage_area': row.get('GarageArea', 500)
                        }
                        
                        try:
                            prediction, _ = process_and_predict(user_inputs)
                            results.append(prediction)
                        except:
                            results.append(np.nan)
                        
                        progress_bar.progress((idx + 1) / len(batch_df))
                
                # Add predictions to dataframe
                batch_df['Predicted_Price'] = results
                batch_df['Price_per_SqFt'] = batch_df['Predicted_Price'] / batch_df.get('GrLivArea', 1500)
                
                st.balloons()
                st.success(f"✅ Predicted {len(results)} houses successfully!")
                
                # Statistics
                st.markdown("### 📊 Batch Prediction Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📈 Mean Price", f"${batch_df['Predicted_Price'].mean():,.0f}")
                with col2:
                    st.metric("📊 Median Price", f"${batch_df['Predicted_Price'].median():,.0f}")
                with col3:
                    st.metric("💎 Min Price", f"${batch_df['Predicted_Price'].min():,.0f}")
                with col4:
                    st.metric("🏆 Max Price", f"${batch_df['Predicted_Price'].max():,.0f}")
                
                # Distribution chart
                fig = px.histogram(
                    batch_df, 
                    x='Predicted_Price',
                    nbins=30,
                    title='Distribution of Predicted Prices',
                    labels={'Predicted_Price': 'Predicted Price ($)'},
                    color_discrete_sequence=['#FF4B4B']
                )
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#FAFAFA" if st.session_state.theme == 'dark' else "#262730"},
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                st.dataframe(
                    batch_df.style.background_gradient(
                        cmap='RdYlGn',
                        subset=['Predicted_Price']
                    ).format({'Predicted_Price': '${:,.0f}', 'Price_per_SqFt': '${:.2f}'}),
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    "📥 DOWNLOAD PREDICTIONS (CSV)",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Make sure your CSV has the required columns")

# ============================================================================
# PAGE 3: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    st.markdown("<h1 style='text-align: center;'>📊 Model Analytics & Insights</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Try to load model comparison data
    try:
        results_df = pd.read_csv('./data/model_comparison_results.csv')
        
        st.markdown("### 🎯 Model Performance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        best = results_df.iloc[0]
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-number'>{best['Model']}</div>
                <p style='color: #888;'>Best Model</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-number'>{best['Val_R2']:.1%}</div>
                <p style='color: #888;'>R² Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-number'>${best['Val_RMSE']:,.0f}</div>
                <p style='color: #888;'>RMSE</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='stat-number'>{metadata['n_features']}</div>
                <p style='color: #888;'>Features</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                results_df.sort_values('Val_RMSE'),
                x='Val_RMSE',
                y='Model',
                orientation='h',
                title='Model Comparison - Validation RMSE',
                color='Val_RMSE',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#FAFAFA" if st.session_state.theme == 'dark' else "#262730"},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                results_df.sort_values('Val_R2', ascending=False),
                x='Val_R2',
                y='Model',
                orientation='h',
                title='Model Comparison - R² Score',
                color='Val_R2',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#FAFAFA" if st.session_state.theme == 'dark' else "#262730"},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        try:
            feat_imp = pd.read_csv('./data/feature_importance.csv')
            
            st.markdown("### 🎯 Top 15 Most Important Features")
            
            fig = px.bar(
                feat_imp.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Features Driving Price Predictions',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#FAFAFA" if st.session_state.theme == 'dark' else "#262730"},
                showlegend=False,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("💡 Feature importance data not available")
    
    except:
        st.warning("⚠️ Model comparison data not found. Run the modeling notebook to generate analytics!")
    
    # Dataset info
    st.markdown("### 📚 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4>📊 Training Data</h4>
            <p><b>Size:</b> 1,460 houses</p>
            <p><b>Features:</b> 231 (after engineering)</p>
            <p><b>Target:</b> Sale Price</p>
            <p><b>Source:</b> Kaggle House Prices Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4>🔧 Preprocessing Steps</h4>
            <p>✅ Data type optimization</p>
            <p>✅ Missing value imputation</p>
            <p>✅ Feature engineering (12+ features)</p>
            <p>✅ Outlier handling</p>
            <p>✅ Categorical encoding</p>
            <p>✅ Feature scaling</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

else:  # About page
    st.markdown("<h1 style='text-align: center;'>ℹ️ About This Application</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🏠 House Price Prediction System
        
        An advanced **Machine Learning application** that predicts house prices with high accuracy using 
        state-of-the-art algorithms and comprehensive data processing.
        
        ### 🎯 Key Features
        
        - **🔮 Single Predictions** - Get instant price estimates for individual houses
        - **📦 Batch Processing** - Upload CSV files to predict multiple houses at once
        - **📊 Visual Analytics** - Explore model performance and feature importance
        - **🎨 Beautiful UI** - Modern, responsive design with dark/light themes
        - **🤖 AI-Powered** - Trained on 1,460+ real estate transactions
        
        ### 🛠️ Technology Stack
        
        - **Frontend:** Streamlit with custom CSS
        - **ML Models:** Scikit-learn (Ridge, Lasso, Random Forest, etc.)
        - **Data Processing:** Pandas, NumPy
        - **Visualizations:** Plotly, interactive charts
        - **Preprocessing:** Custom pipeline with feature engineering
        
        ### 📊 Model Performance
        
        - **Accuracy:** {:.1%} (R² Score)
        - **Average Error:** ${:,.0f} (RMSE)
        - **Features Used:** {} engineered features
        - **Training Data:** 1,460 houses
        - **Validation Split:** 80/20 train-test
        
        ### 🎓 How It Works
        
        1. **Input Collection** - You provide house characteristics
        2. **Preprocessing** - Data is cleaned, scaled, and encoded
        3. **Feature Engineering** - 12+ new features created
        4. **ML Prediction** - Trained model estimates price
        5. **Results Display** - Beautiful visualization of prediction
        
        ### 💡 Use Cases
        
        - **Home Buyers** - Estimate fair price before making an offer
        - **Real Estate Agents** - Quick valuation tool for clients
        - **Investors** - Analyze property values in bulk
        - **Developers** - Understand price drivers for new projects
        
        ### 📈 Future Enhancements
        
        - [ ] Add more ML models (XGBoost, LightGBM)
        - [ ] Implement SHAP values for interpretability
        - [ ] Add neighborhood-level analytics
        - [ ] Historical price trend predictions
        - [ ] Mobile app version
        
        """.format(
            metadata['val_r2'] if metadata else 0.89,
            metadata['val_rmse'] if metadata else 22000,
            metadata['n_features'] if metadata else 231
        ))
    
    with col2:
        st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <div style='font-size: 4rem; margin: 2rem 0;'>🏆</div>
            <h3>Award-Winning Accuracy</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #FF4B4B;'>{:.1%}</p>
            <p>Prediction Accuracy</p>
        </div>
        
        <div class='info-card' style='text-align: center; margin-top: 2rem;'>
            <div style='font-size: 4rem; margin: 2rem 0;'>📊</div>
            <h3>Comprehensive Analysis</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #FF4B4B;'>{}</p>
            <p>Features Analyzed</p>
        </div>
        
        <div class='info-card' style='text-align: center; margin-top: 2rem;'>
            <div style='font-size: 4rem; margin: 2rem 0;'>⚡</div>
            <h3>Lightning Fast</h3>
            <p style='font-size: 2rem; font-weight: 700; color: #FF4B4B;'>&lt;2s</p>
            <p>Prediction Time</p>
        </div>
        """.format(
            metadata['val_r2'] if metadata else 0.89,
            metadata['n_features'] if metadata else 231
        ), unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📧 Contact & Support
        
        - 📖 [Documentation](#)
        - 💬 zeinab.h.mahfouz@gmail.com
        - ⭐ https://github.com/ZeinabMahfouz
        - 📧 zeinab.h.mahfouz@gmail.com
        """)
        
        st.markdown("---")
        
        # Rating
        st.markdown("### ⭐ Rate This App")
        rating = st.slider("How satisfied are you?", 1, 5, 5)
        
        if rating >= 4:
            st.success("🎉 Thank you! We're glad you love it!")
        elif rating == 3:
            st.info("😊 Thanks! We'll keep improving!")
        else:
            st.warning("😔 Sorry! Please share feedback to help us improve.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1.2rem;'><b>🏠 House Price Predictor</b></p>
    <p>Built with ❤️ using Streamlit, Scikit-learn & ML</p>
    <p>© 2024 All Rights Reserved | Powered by AI</p>
</div>
""", unsafe_allow_html=True)