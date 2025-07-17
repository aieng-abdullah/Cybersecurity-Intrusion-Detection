import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Step 1: Load raw dataset
df = pd.read_csv(r'C:\Users\teamp\Desktop\Cybersecurity Intrusion Detection\data\raw\cybersecurity_intrusion_data.csv')
logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Step 2: Define columns for preprocessing
drop_cols = ['session_id']  # Columns to drop (non-informative IDs)
cat_cols = ['protocol_type', 'encryption_used', 'browser_type']  # Categorical features
num_cols = ['network_packet_size', 'login_attempts', 'session_duration',
            'ip_reputation_score', 'failed_logins', 'unusual_time_access']  # Numerical features

# Step 3: Build preprocessing pipelines
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),  # Fill missing categorical with 'Missing'
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical variables
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing numerical with mean
    ('scaler', StandardScaler())  # Standard scale numerical features
])

# Combine pipelines into a ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')  # Drop other columns not listed explicitly

# Step 4: Separate features and target
X = df.drop(columns=drop_cols + ['attack_detected'])
y = df['attack_detected']

# Step 5: Fit and transform the features
X_processed = preprocessor.fit_transform(X)
logger.info(f"Preprocessing complete. Processed data shape: {X_processed.shape}")

# Step 6: Get transformed categorical feature names
cat_features_encoded = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
all_features = num_cols + list(cat_features_encoded)

# Convert processed array back to a DataFrame with proper feature names
X_processed_df = pd.DataFrame(X_processed, columns=all_features)

# Add target column back to the processed DataFrame
X_processed_df['attack_detected'] = y.values

# Step 7: Ensure the save directory exists
save_dir = 'data/preprocess'
os.makedirs(save_dir, exist_ok=True)

# Step 8: Save the preprocessed data as CSV
save_path = os.path.join(save_dir, 'preprocessed_data.csv')
X_processed_df.to_csv(save_path, index=False)
logger.info(f"Preprocessed data saved to {save_path}")
