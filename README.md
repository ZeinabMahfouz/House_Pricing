# 🏠 House Price Prediction - Complete ML Pipeline


A comprehensive **end-to-end Machine Learning project** for predicting house prices with an interactive web application. Features advanced data preprocessing, multiple ML models with hyperparameter tuning, and a beautiful Streamlit dashboard.



---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Models](#-models)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🎯 Machine Learning
- **8 Regression Models**: Linear, Polynomial, Ridge, Lasso, ElasticNet, Decision Tree, SVR, KNN
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold cross-validation
- **Feature Engineering**: 12+ engineered features for better predictions
- **Dimensionality Reduction**: Multiple PCA configurations tested
- **Model Validation**: Train/validation split with comprehensive metrics

### 🔧 Data Processing
- **Smart Imputation**: Context-aware missing value handling
- **Data Type Optimization**: 30-40% memory reduction
- **Outlier Handling**: IQR-based detection and capping
- **Skewness Transformation**: Log transformation for normalized distributions
- **Categorical Encoding**: Ordinal + One-hot encoding
- **Feature Scaling**: RobustScaler for outlier-resistant normalization

### 🎨 Interactive Web App
- **Dark/Light Mode**: Toggle between themes
- **Real-time Predictions**: Instant price estimates
- **Data Analysis Dashboard**: Comprehensive visualizations
- **Feature Explanations**: Understand what drives prices
- **Sample Data**: Pre-loaded test cases
- **Responsive Design**: Works on all devices

---

## 🎥 Demo

### Live Demo
🔗 [[Try the app here](https://your-app-url.streamlit.app)](https://housepricing-zeinabmahfouz.streamlit.app/)

### Screenshots

| Prediction Page | Data Analysis | Feature Guide |
|---|---|---|
| ![Predict](images/predict_page.png) | ![Analysis](images/analysis_page.png) | ![Guide](images/feature_guide.png) |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Clone the Repository
```bash
git clone https://github.com/yourusername/house-pricing.git
cd house-pricing
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Data
1. Go to [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Download `train.csv` and `test.csv`
3. Place them in `data/raw/`

---

## 📁 Project Structure

```
House_Pricing/
│
├── data/
│   ├── raw/                      # Original datasets
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/                # Preprocessed data
│   │   ├── train_processed.csv
│   │   ├── test_processed.csv
│   │   └── train_pca_*.csv
│   ├── artifacts/                # Saved transformers
│   │   ├── statistics.pkl
│   │   ├── scaler.pkl
│   │   └── pca_models.pkl
│   ├── feature_descriptions.csv  # Feature explanations
│   └── sample_houses.csv         # Test samples
│
├── notebooks/
│   ├── 01_profiling.ipynb        # Data exploration
│   ├── 02_preprocessing.ipynb    # Data preprocessing
│   └── 03_modeling_comprehensive.ipynb  # Model training
│
├── src/
│   └── preprocessing.py          # Preprocessing module
│
├── models/
│   ├── best_model.pkl            # Trained model
│   └── model_metadata.pkl        # Model info
│
├── images/                       # Visualizations
│   ├── preprocessing_summary.png
│   ├── comprehensive_model_comparison.png
│   └── app_screenshot.png
│
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## 💻 Usage

### 1. Data Profiling
Generate comprehensive data analysis report:

```bash
jupyter notebook notebooks/01_profiling.ipynb
```

This creates `data/profiling_report.html` with:
- Missing value patterns
- Variable distributions
- Correlations
- Data quality warnings

### 2. Data Preprocessing
Run the preprocessing pipeline:

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**What it does:**
- ✅ Optimizes data types (saves 30-40% memory)
- ✅ Handles missing values intelligently
- ✅ Engineers 12+ new features
- ✅ Detects and caps outliers
- ✅ Transforms skewed distributions
- ✅ Encodes categorical variables
- ✅ Scales features
- ✅ Creates multiple PCA versions
- ✅ Saves artifacts for test data

**Output:**
- `train_processed.csv` - Ready for modeling
- `test_processed.csv` - Ready for predictions
- `train_pca_*.csv` - PCA versions for comparison

### 3. Model Training
Train and compare multiple models:

```bash
jupyter notebook notebooks/03_modeling_comprehensive.ipynb
```

**Models trained:**
1. Linear Regression
2. Polynomial Regression (degree 2-3)
3. Ridge Regression
4. Lasso Regression
5. ElasticNet
6. Decision Tree Regressor
7. Support Vector Regressor (SVR)
8. K-Nearest Neighbors (KNN)

Each model undergoes:
- ✅ Hyperparameter tuning (RandomizedSearchCV)
- ✅ 5-fold cross-validation
- ✅ Train/validation evaluation
- ✅ Overfitting analysis

**Output:**
- `best_model.pkl` - Best performing model
- `model_metadata.pkl` - Model information
- `model_comparison_results.csv` - All results
- `feature_importance.csv` - Feature rankings

### 4. Run the Web App
Launch the interactive dashboard:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

**App Features:**
- 🏠 **Predict Price**: Enter house details, get instant estimates
- 📊 **Data Analysis**: Interactive visualizations and insights
- 📖 **Feature Guide**: Understand all features
- 🎯 **Sample Data**: Test with pre-loaded examples
- ℹ️ **About**: Project information

---

## 🔄 Data Pipeline

### Input Data
- **Training**: 1,460 houses with 81 features
- **Test**: 1,459 houses with 80 features (no price)

### Data Type Optimization
```python
# Memory-efficient conversions
MSSubClass: int64 → category   # Building type code
MoSold: int64 → category        # Month (no ordinal meaning)
OverallQual: int64 → int8       # 1-10 scale
YearBuilt: int64 → int16        # Years (1800-2100)
LotArea: int64 → float32        # Square footage
```
**Result**: 30-40% memory reduction

### Missing Value Handling

| Strategy | Features | Reason |
|---|---|---|
| Fill with "None" | PoolQC, Fence, Alley, Garage*, Bsmt* | NA = Feature doesn't exist |
| Fill with Mode | MSZoning, Exterior*, Electrical | Most common value |
| Fill with Median | LotFrontage, MasVnrArea | Numeric imputation |
| Fill with 0 | Area measurements | No area = 0 |
| Neighborhood Median | LotFrontage | Location-based |

### Feature Engineering

| New Feature | Formula | Purpose |
|---|---|---|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | Total living space |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | Total bathrooms |
| `HouseAge` | YrSold - YearBuilt | Age of house |
| `RemodAge` | YrSold - YearRemodAdd | Years since remodel |
| `TotalPorchSF` | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | Total porch area |
| `HasPool` | PoolArea > 0 | Binary indicator |
| `Has2ndFloor` | 2ndFlrSF > 0 | Binary indicator |
| `HasGarage` | GarageArea > 0 | Binary indicator |
| `HasBsmt` | TotalBsmtSF > 0 | Binary indicator |
| `HasFireplace` | Fireplaces > 0 | Binary indicator |
| `OverallQualCond` | OverallQual × OverallCond | Quality interaction |

### Imbalanced Feature Handling

| Action | Features | Reason |
|---|---|---|
| **Drop** | Street (99.6%), Utilities (99.9%) | No variation |
| **Binary Encode** | LandSlope, RoofMatl, Heating | High imbalance (95-99%) |
| **Keep** | Most others | Acceptable balance |

### Transformations

1. **Outlier Capping**: IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
2. **Skewness**: Log1p for features with |skewness| > 0.75
3. **Encoding**: Ordinal for quality features, One-hot for nominal
4. **Scaling**: RobustScaler (resistant to outliers)

---

## 🤖 Models

### Model Comparison

| Model | Validation RMSE | Validation R² | Train/Val Gap | Notes |
|---|---|---|---|---|
| **Ridge** | **$22,145** | **0.891** | 0.034 | ✅ Best overall |
| Lasso | $23,089 | 0.881 | 0.029 | Good feature selection |
| ElasticNet | $23,456 | 0.877 | 0.031 | Balanced regularization |
| Linear Regression | $24,012 | 0.871 | 0.048 | Baseline |
| Decision Tree | $26,234 | 0.845 | 0.112 | Overfits |
| Polynomial (deg=2) | $25,678 | 0.852 | 0.089 | Some overfitting |
| SVR | $28,901 | 0.821 | 0.045 | Slower training |
| KNN | $30,456 | 0.798 | 0.067 | Distance-based |

### Hyperparameter Tuning Results

**Best Ridge Model:**
```python
{
    'alpha': 10.0,
    'solver': 'cholesky'
}
```

**Best Decision Tree:**
```python
{
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt'
}
```

### PCA Impact

| Version | Components | Variance Explained | RMSE | Impact |
|---|---|---|---|---|
| Original | 231 | 100% | $22,145 | **Best** |
| PCA 99% | 203 | 99% | $22,567 | -$422 |
| PCA 95% | 156 | 95% | $23,890 | -$1,745 |
| PCA 100 | 100 | 87% | $25,123 | -$2,978 |

**Conclusion**: Original features perform best. PCA only if speed is critical.

---

## 📊 Results

### Model Performance

- **Best Model**: Ridge Regression
- **Validation RMSE**: $22,145
- **Validation R²**: 0.891 (89.1% variance explained)
- **Validation MAE**: $15,234
- **Overfitting**: Minimal (3.4% gap between train/val R²)

### Top 10 Most Important Features

1. **OverallQual** (0.124) - Overall material and finish quality
2. **GrLivArea** (0.089) - Above ground living area
3. **TotalSF** (0.076) - Total square footage (engineered)
4. **GarageCars** (0.058) - Garage capacity
5. **TotalBath** (0.054) - Total bathrooms (engineered)
6. **YearBuilt** (0.047) - Construction year
7. **1stFlrSF** (0.042) - First floor area
8. **GarageArea** (0.039) - Garage square footage
9. **TotalBsmtSF** (0.037) - Basement area
10. **OverallQualCond** (0.033) - Quality × Condition (engineered)

### Feature Correlations with Price

**Strongest Positive:**
- OverallQual: +0.79
- GrLivArea: +0.71
- GarageCars: +0.64
- TotalBath: +0.63
- YearBuilt: +0.52

**Strongest Negative:**
- KitchenAbvGr: -0.14
- OverallCond: -0.08
- YrSold: -0.03

---

## 🎨 Streamlit App Features

### 🏠 Prediction Page
- **Interactive Form**: Enter house characteristics
- **Sample Data Loading**: Quick start with pre-loaded examples
- **Real-time Estimation**: Instant price prediction
- **Price Visualization**: Animated gauge chart
- **Confidence Interval**: Uncertainty estimate
- **Market Comparison**: Compare with average prices
- **Detailed Metrics**: Price per sq ft, house age, total area

### 📊 Data Analysis Page
- **Model Comparison**: Interactive bar charts
- **Overfitting Analysis**: Train vs validation comparison
- **Feature Importance**: Top features visualization
- **Price Distribution**: Histograms (original & log-transformed)
- **Correlation Analysis**: Positive and negative correlations
- **Interactive Filtering**: Explore data dynamically

### 📖 Feature Guide
- **Searchable**: Find features quickly
- **Category Filtering**: Filter by building, quality, size, etc.
- **Detailed Descriptions**: What each feature means
- **Examples**: Sample values
- **Importance Ratings**: High/Medium/Low

### 🎯 Sample Data
- **10 Pre-loaded Houses**: From starter to mansion
- **Statistics**: Average metrics across samples
- **One-click Loading**: Test instantly
- **Data Table**: View all sample details

### Theme Support
- **Dark Mode**: Default, easy on eyes
- **Light Mode**: Toggle with one click
- **Persistent**: Remembers your preference
- **Smooth Transitions**: Animated theme changes

---

## 🛠️ Technical Details

### Dependencies

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
matplotlib==3.7.2
seaborn==0.12.2
ydata-profiling==4.5.1
streamlit==1.28.0
plotly==5.17.0
```

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for data and models
- **CPU**: Any modern processor
- **OS**: Windows, macOS, or Linux

---

## 📈 Future Improvements

- [ ] Add XGBoost, LightGBM, CatBoost models
- [ ] Implement stacking/ensemble methods
- [ ] Add SHAP values for interpretability
- [ ] Create interactive map visualization
- [ ] Add historical price trends
- [ ] Implement user authentication
- [ ] Add prediction history tracking
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Create REST API
- [ ] Add automated retraining pipeline

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/house-pricing.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for the House Prices dataset
- **Streamlit** for the amazing web framework
- **Scikit-learn** for ML algorithms
- **Plotly** for interactive visualizations
- All contributors and supporters

---

## 📧 Contact

**Your Name**
- 📧 Email: zeinab.h.mahfouz@gmail.com
- 💼 LinkedIn: https://www.linkedin.com/in/zeinab-mahfouz/
- 🐙 GitHub: https://github.com/ZeinabMahfouz

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/house-pricing&type=Date)](https://star-history.com/#yourusername/house-pricing&Date)

---

<div align="center">
  <p>Made with ❤️ and lots of ☕</p>
  <p>If you found this project helpful, please consider giving it a ⭐</p>
</div>
