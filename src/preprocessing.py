
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class HousePricePreprocessor:
    """
    Complete preprocessing pipeline for House Prices dataset.
    Saves all statistics and transformers for consistent test data processing.
    """
    
    def __init__(self, artifacts_path='../data/artifacts/'):
        self.artifacts_path = artifacts_path
        self.statistics = {}
        self.encoders = {}
        self.scaler = None
        self.pca_models = {}
        self.dtype_mappings = {}
    
    # ========================================================================
    # DATA TYPE OPTIMIZATION
    # ========================================================================
    
    def optimize_data_types(self, df, is_train=True):
        """
        Optimize data types for memory efficiency and correct semantics
        IMPORTANT: Call this FIRST in the pipeline
        """
        print("\n" + "=" * 80)
        print("DATA TYPE OPTIMIZATION")
        print("=" * 80)
        
        df = df.copy()
        
        # Memory before
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        if is_train:
            # Define optimal data types
            dtype_mappings = {
                # Categorical codes (not continuous numbers)
                'MSSubClass': 'str',  # Dwelling type code
                'MoSold': 'str',      # Month sold
                
                # Ordinal scales (1-10)
                'OverallQual': 'int8',
                'OverallCond': 'int8',
                
                # Years (int16 sufficient for 1800-2100)
                'YearBuilt': 'int16',
                'YearRemodAdd': 'int16',
                'GarageYrBlt': 'float32',  # Has NaN, needs float
                'YrSold': 'int16',
                
                # Count fields (0-255 range)
                'BsmtFullBath': 'int8',
                'BsmtHalfBath': 'int8',
                'FullBath': 'int8',
                'HalfBath': 'int8',
                'BedroomAbvGr': 'int8',
                'KitchenAbvGr': 'int8',
                'TotRmsAbvGrd': 'int8',
                'Fireplaces': 'int8',
                'GarageCars': 'float32',  # Has NaN in test set
                
                # Area measurements (float32 sufficient)
                'LotFrontage': 'float32',
                'LotArea': 'float32',
                'MasVnrArea': 'float32',
                'BsmtFinSF1': 'float32',
                'BsmtFinSF2': 'float32',
                'BsmtUnfSF': 'float32',
                'TotalBsmtSF': 'float32',
                '1stFlrSF': 'float32',
                '2ndFlrSF': 'float32',
                'LowQualFinSF': 'float32',
                'GrLivArea': 'float32',
                'GarageArea': 'float32',
                'WoodDeckSF': 'float32',
                'OpenPorchSF': 'float32',
                'EnclosedPorch': 'float32',
                '3SsnPorch': 'float32',
                'ScreenPorch': 'float32',
                'PoolArea': 'float32',
                'MiscVal': 'float32',
            }
            
            self.dtype_mappings = dtype_mappings
        else:
            # Use saved dtype mappings
            dtype_mappings = self.dtype_mappings
        
        # Apply dtype conversions
        changes_made = 0
        for col, dtype in dtype_mappings.items():
            if col in df.columns:
                try:
                    old_dtype = df[col].dtype
                    df[col] = df[col].astype(dtype)
                    changes_made += 1
                    print(f"✓ {col:20s}: {str(old_dtype):10s} → {dtype}")
                except Exception as e:
                    print(f"⚠️  Could not convert {col}: {e}")
        
        # Memory after
        memory_after = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = memory_before - memory_after
        memory_saved_pct = (memory_saved / memory_before) * 100 if memory_before > 0 else 0
        
        print(f"\n📊 Memory: {memory_before:.2f} MB → {memory_after:.2f} MB")
        print(f"💾 Saved: {memory_saved:.2f} MB ({memory_saved_pct:.1f}%)")
        print(f"✓ Optimized {changes_made} columns")
        
        return df
    
    # ========================================================================
    # DATA TYPE ANALYSIS
    # ========================================================================
    
    def analyze_data_types(self, df):
        """Analyze current data types"""
        print("\n" + "=" * 80)
        print("DATA TYPE SUMMARY")
        print("=" * 80)
        
        dtype_counts = df.dtypes.value_counts()
        print("\nData type distribution:")
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        return df
    
    # ========================================================================
    # MISSING VALUE HANDLING
    # ========================================================================
    
    def detect_missing_values(self, df):
        """Comprehensive missing value analysis"""
        print("\n" + "=" * 80)
        print("MISSING VALUE ANALYSIS")
        print("=" * 80)
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        )
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            print(f"\nTotal columns with missing values: {len(missing_df)}")
        else:
            print("No missing values found!")
        
        return missing_df
    
    def impute_missing_values(self, df, is_train=True):
        """
        Intelligent missing value imputation
        For train: calculate and save statistics
        For test: use saved statistics
        """
        print("\n" + "=" * 80)
        print("MISSING VALUE IMPUTATION")
        print("=" * 80)
        
        df = df.copy()
        
        # Features where NA means "None/Not Applicable"
        none_features = [
            'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
            'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'
        ]
        
        # Features to impute with mode
        mode_features = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                         'MasVnrType', 'Electrical', 'KitchenQual', 'Functional',
                         'SaleType']
        
        # Features to impute with median
        median_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        
        # Features to impute with 0
        zero_features = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                         'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
        
        # Impute "None"
        for col in none_features:
            if col in df.columns:
                df[col].fillna('None', inplace=True)
                print(f"✓ {col}: Filled with 'None'")
        
        # Impute with mode
        for col in mode_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                if is_train:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    self.statistics[f'{col}_mode'] = mode_val
                else:
                    mode_val = self.statistics.get(f'{col}_mode', 'Unknown')
                
                df[col].fillna(mode_val, inplace=True)
                print(f"✓ {col}: Filled with mode '{mode_val}'")
        
        # Impute with median
        for col in median_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                if is_train:
                    median_val = df[col].median()
                    self.statistics[f'{col}_median'] = median_val
                else:
                    median_val = self.statistics.get(f'{col}_median', df[col].median())
                
                df[col].fillna(median_val, inplace=True)
                print(f"✓ {col}: Filled with median {median_val:.2f}")
        
        # Impute with 0
        for col in zero_features:
            if col in df.columns:
                df[col].fillna(0, inplace=True)
                print(f"✓ {col}: Filled with 0")
        
        # Special: LotFrontage by neighborhood
        if 'LotFrontage' in df.columns and df['LotFrontage'].isnull().sum() > 0:
            if is_train:
                neighborhood_medians = df.groupby('Neighborhood')['LotFrontage'].median()
                self.statistics['neighborhood_lotfrontage'] = neighborhood_medians
            else:
                neighborhood_medians = self.statistics.get('neighborhood_lotfrontage', pd.Series())
            
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
            print("✓ LotFrontage: Filled with neighborhood median")
        
        return df
    
    # ========================================================================
    # IMBALANCED FEATURE HANDLING
    # ========================================================================
    
    def handle_imbalanced_features(self, df, is_train=True):
        """Handle severely imbalanced categorical features"""
        print("\n" + "=" * 80)
        print("IMBALANCED FEATURE HANDLING")
        print("=" * 80)
        
        df = df.copy()
        
        # Drop extremely imbalanced features (>99% one class)
        drop_features = ['Street', 'Utilities']
        
        for col in drop_features:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                print(f"✓ Dropped '{col}' (>99% imbalanced)")
        
        # Binary conversion for high imbalance
        if is_train:
            binary_conversions = {}
            
            if 'LandSlope' in df.columns:
                binary_conversions['LandSlope'] = 'Gtl'
            
            if 'RoofMatl' in df.columns:
                binary_conversions['RoofMatl'] = 'CompShg'
            
            if 'Heating' in df.columns:
                binary_conversions['Heating'] = 'GasA'
            
            self.statistics['binary_conversions'] = binary_conversions
        else:
            binary_conversions = self.statistics.get('binary_conversions', {})
        
        for col, dominant_value in binary_conversions.items():
            if col in df.columns:
                new_col_name = f'Is_{dominant_value}'
                df[new_col_name] = (df[col] == dominant_value).astype(int)
                df.drop(col, axis=1, inplace=True)
                print(f"✓ Converted '{col}' to binary '{new_col_name}'")
        
        return df
    
    # ========================================================================
    # FEATURE ENGINEERING
    # ========================================================================
    
    def feature_engineering(self, df):
        """Create new meaningful features"""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        df = df.copy()
        
        # Total square footage
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        print("✓ Created TotalSF")
        
        # Total bathrooms
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + 
                           df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        print("✓ Created TotalBath")
        
        # House age
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        print("✓ Created HouseAge and RemodAge")
        
        # Total porch area
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + 
                              df['3SsnPorch'] + df['ScreenPorch'])
        print("✓ Created TotalPorchSF")
        
        # Binary features
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        print("✓ Created binary features")
        
        # Quality interaction
        df['OverallQualCond'] = df['OverallQual'] * df['OverallCond']
        print("✓ Created OverallQualCond")
        
        return df
    
    # ========================================================================
    # OUTLIER HANDLING
    # ========================================================================
    
    def detect_outliers(self, df, z_threshold=3):
        """Detect outliers using IQR method"""
        print("\n" + "=" * 80)
        print("OUTLIER DETECTION")
        print("=" * 80)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = []
        
        for col in numeric_cols:
            if col == 'Id':
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = len(df[(df[col] < Q1 - 1.5 * IQR) | 
                                   (df[col] > Q3 + 1.5 * IQR)])
            
            if iqr_outliers > 0:
                outlier_summary.append({
                    'Feature': col,
                    'IQR_outliers': iqr_outliers
                })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            print(outlier_df.to_string(index=False))
        else:
            print("No outliers detected")
        
        return outlier_summary
    
    def handle_outliers(self, df, is_train=True, method='cap'):
        """Handle outliers using capping"""
        print("\n" + "=" * 80)
        print(f"OUTLIER HANDLING (Method: {method.upper()})")
        print("=" * 80)
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
        
        for col in numeric_cols:
            if is_train:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.statistics[f'{col}_lower_bound'] = lower_bound
                self.statistics[f'{col}_upper_bound'] = upper_bound
            else:
                lower_bound = self.statistics.get(f'{col}_lower_bound', df[col].min())
                upper_bound = self.statistics.get(f'{col}_upper_bound', df[col].max())
            
            if method == 'cap':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"✓ Outliers handled for {len(numeric_cols)} features")
        return df
    
    # ========================================================================
    # SKEWNESS TRANSFORMATION
    # ========================================================================
    
    def transform_skewed_features(self, df, is_train=True, skew_threshold=0.75):
        """Apply log transformation to highly skewed features"""
        print("\n" + "=" * 80)
        print("SKEWNESS TRANSFORMATION")
        print("=" * 80)
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
        
        if is_train:
            skewed_features = []
            for col in numeric_cols:
                skewness = df[col].skew()
                if abs(skewness) > skew_threshold:
                    skewed_features.append(col)
            
            self.statistics['skewed_features'] = skewed_features
            print(f"Found {len(skewed_features)} skewed features")
        else:
            skewed_features = self.statistics.get('skewed_features', [])
        
        # Apply log1p transformation
        for col in skewed_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        
        print(f"✓ Applied log transformation to {len(skewed_features)} features")
        return df
    
    # ========================================================================
    # CATEGORICAL ENCODING
    # ========================================================================
    
    def encode_categorical_features(self, df, is_train=True):
        """Encode categorical features"""
        print("\n" + "=" * 80)
        print("CATEGORICAL ENCODING")
        print("=" * 80)
        
        df = df.copy()
        
        # Ordinal features with inherent order
        ordinal_mappings = {
            'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'PoolQC': {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
            'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        }
        
        # Apply ordinal encoding
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                print(f"✓ {col}: Ordinal encoded")
        
        # One-hot encoding for remaining categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Id']
        
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            print(f"✓ One-hot encoded {len(categorical_cols)} features")
        
        return df
    
    # ========================================================================
    # FEATURE SCALING
    # ========================================================================
    
    def scale_features(self, df, is_train=True, method='robust'):
        """Scale numeric features"""
        print("\n" + "=" * 80)
        print(f"FEATURE SCALING (Method: {method.upper()})")
        print("=" * 80)
        
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in ['Id', 'SalePrice']]
        
        if is_train:
            if method == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
            
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            if self.scaler is not None:
                df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        
        print(f"✓ Scaled {len(cols_to_scale)} features")
        return df
   
    def apply_pca(self, df, variance_thresholds=[0.95, 0.99], n_components_list=[50, 100, 150]):
        """Apply PCA with different configurations"""
        print("\n" + "=" * 80)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 80)
        
        df_features = df.drop(['Id', 'SalePrice'], axis=1, errors='ignore')
        pca_results = {}
        
        # PCA by variance
        for var_threshold in variance_thresholds:
            pca = PCA(n_components=var_threshold, random_state=42)
            pca_features = pca.fit_transform(df_features)
            
            n_components = pca.n_components_
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            if 'Id' in df.columns:
                pca_df['Id'] = df['Id'].values
            if 'SalePrice' in df.columns:
                pca_df['SalePrice'] = df['SalePrice'].values
            
            pca_results[f'variance_{int(var_threshold*100)}'] = pca_df
            self.pca_models[f'variance_{int(var_threshold*100)}'] = pca
            
            print(f"✓ PCA (variance={var_threshold}): {n_components} components")
        
        # PCA by fixed components
        for n_comp in n_components_list:
            if n_comp <= df_features.shape[1]:
                pca = PCA(n_components=n_comp, random_state=42)
                pca_features = pca.fit_transform(df_features)
                
                pca_df = pd.DataFrame(
                    pca_features,
                    columns=[f'PC{i+1}' for i in range(n_comp)]
                )
                
                if 'Id' in df.columns:
                    pca_df['Id'] = df['Id'].values
                if 'SalePrice' in df.columns:
                    pca_df['SalePrice'] = df['SalePrice'].values
                
                pca_results[f'n_components_{n_comp}'] = pca_df
                self.pca_models[f'n_components_{n_comp}'] = pca
                
                print(f"✓ PCA (n={n_comp}): explained variance = {pca.explained_variance_ratio_.sum():.4f}")
        
        return pca_results
    
    
    def save_artifacts(self):
        """Save all statistics, scalers, and models"""
        print("\n" + "=" * 80)
        print("SAVING ARTIFACTS")
        print("=" * 80)
        
        os.makedirs(self.artifacts_path, exist_ok=True)
        
        # Save statistics and dtype mappings
        artifacts_to_save = {
            'statistics': self.statistics,
            'dtype_mappings': self.dtype_mappings
        }
        
        with open(f'{self.artifacts_path}statistics.pkl', 'wb') as f:
            pickle.dump(artifacts_to_save, f)
        print(f"✓ Saved statistics & dtype mappings")
        
        # Save scaler
        if self.scaler is not None:
            with open(f'{self.artifacts_path}scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Saved scaler")
        
        # Save PCA models
        if self.pca_models:
            with open(f'{self.artifacts_path}pca_models.pkl', 'wb') as f:
                pickle.dump(self.pca_models, f)
            print(f"✓ Saved PCA models")
    
    def load_artifacts(self):
        """Load all saved artifacts"""
        print("\n" + "=" * 80)
        print("LOADING ARTIFACTS")
        print("=" * 80)
        
        # Load statistics and dtype mappings
        with open(f'{self.artifacts_path}statistics.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        # Handle old format vs new format
        if 'statistics' in artifacts:
            self.statistics = artifacts['statistics']
            self.dtype_mappings = artifacts.get('dtype_mappings', {})
        else:
            self.statistics = artifacts
            self.dtype_mappings = {}
        
        print(f"✓ Loaded statistics")
        
        # Load scaler
        try:
            with open(f'{self.artifacts_path}scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Loaded scaler")
        except:
            print("⚠ Scaler not found")
        
        # Load PCA models
        try:
            with open(f'{self.artifacts_path}pca_models.pkl', 'rb') as f:
                self.pca_models = pickle.load(f)
            print(f"✓ Loaded PCA models")
        except:
            print("⚠ PCA models not found")
    
    def preprocess_pipeline(self, df, is_train=True, apply_pca_flag=False):
        """
        Complete preprocessing pipeline
        """
        print("\n" + "=" * 80)
        print(f"{'TRAINING' if is_train else 'TEST'} DATA PREPROCESSING PIPELINE")
        print("=" * 80)
        print(f"Initial shape: {df.shape}")
        
        # 0. Optimize data types (FIRST!)
        df = self.optimize_data_types(df, is_train=is_train)
        
        # 1. Data type analysis
        df = self.analyze_data_types(df)
        
        # 2. Missing value detection and imputation
        if is_train:
            self.detect_missing_values(df)
        df = self.impute_missing_values(df, is_train=is_train)
        
        # 3. Handle imbalanced features
        df = self.handle_imbalanced_features(df, is_train=is_train)
        
        # 4. Feature engineering
        df = self.feature_engineering(df)
        
        # 5. Outlier detection and handling
        if is_train:
            self.detect_outliers(df)
        df = self.handle_outliers(df, is_train=is_train, method='cap')
        
        # 6. Transform skewed features
        df = self.transform_skewed_features(df, is_train=is_train)
        
        # 7. Encode categorical features
        df = self.encode_categorical_features(df, is_train=is_train)
        
        # 8. Scale features
        df = self.scale_features(df, is_train=is_train, method='robust')
        
        # 9. Apply PCA if requested
        pca_results = None
        if apply_pca_flag and is_train:
            pca_results = self.apply_pca(df)
        
        # Save artifacts if training
        if is_train:
            self.save_artifacts()
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Final shape: {df.shape}")
        
        return df, pca_results


if __name__ == "__main__":
    import pandas as pd
    import os
    PROCESSED_DIR = '../data/processed/'
    ARTIFACTS_DIR = '../data/artifacts/' 
    os.makedirs(PROCESSED_DIR, exist_ok=True) 
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)  
    # Initialize preprocessor
    preprocessor = HousePricePreprocessor(artifacts_path='../data/artifacts/')
        
    # Load data
    print("Loading data...")
    train_df_raw = pd.read_csv(r'C:\Users\Zeina\House_Pricing\data\raw\train.csv')
    test_df = pd.read_csv(r'C:\Users\Zeina\House_Pricing\data\raw\test.csv')
    print(f"Train: {train_df_raw.shape}")
    print(f"Test: {test_df.shape}")
    print("\n" + "="*80)
    print("PROCESSING TRAINING DATA")
    print("="*80)
    y_train = np.log1p(train_df_raw['SalePrice'])
    X_train_raw = train_df_raw.drop('SalePrice', axis=1) 
    train_df = X_train_raw.copy()
    train_df['SalePrice'] = y_train  
    train_processed, pca_results = preprocessor.preprocess_pipeline(
            train_df,
            is_train=True,
            apply_pca_flag=True
        )
    
     # Save processed training data
    train_processed.to_csv('../data/processed/train_processed.csv', index=False)
    print("\n✓ Saved: train_processed.csv")
     # Save PCA versions
    if pca_results:
        for version_name, pca_df in pca_results.items():
            filename = f'../data/processed/train_pca_{version_name}.csv'
            pca_df.to_csv(filename, index=False)
            print(f"✓ Saved: {filename}")
        
        # Process test data
    print("\n" + "="*80)
    print("PROCESSING TEST DATA")
    print("="*80)
    preprocessor.load_artifacts()
    test_processed, _ = preprocessor.preprocess_pipeline(
            test_df,
            is_train=False,
            apply_pca_flag=False
        )
        
    # Save processed test data
    test_processed.to_csv('../data/processed/test_processed.csv', index=False)
    print("\n✓ Saved: test_processed.csv")
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"Train features: {train_processed.shape[1] - 2}")
    print(f"Test features: {test_processed.shape[1] - 1}")
    print(f"Alignment OK: {train_processed.shape[1] - 1 == test_processed.shape[1]}")
        
    print("\n✅ PREPROCESSING COMPLETE!")
    print("\nNext steps:")
    print("1. Train models on: data/processed/train_processed.csv")
    print("2. Compare PCA versions if needed")
    print("3. Generate predictions on: data/processed/test_processed.csv")