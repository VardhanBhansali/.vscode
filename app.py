import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import glob
import os
import warnings
import re
from datetime import timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Functions ---

def normalize_column_names(df):
    """
    Normalize column names by converting to uppercase, removing whitespace, and replacing special characters with underscores.
    """
    df.columns = (df.columns
                  .str.strip()              # Remove leading/trailing whitespace
                  .str.upper()              # Convert to uppercase
                  .str.replace(r'[^\w\s]', '')  # Remove special characters
                  .str.replace(r'\s+', '_')     # Replace spaces with underscores
                  )
    return df

def build_prediction_model(data):
    try:
        # Required columns (UQC removed)
        required_cols = ['QUANTITY', 'ITEM_RATE_INR', 'HSCODE', 'FOREIGNPORT', 'ITEM_RATE_INV', 
                        'TOTAL_AMOUNT_INV_FC', 'FOREIGNCOUNTRY', 'INDIANPORT', 'CURRENCY']
        numeric_features = ['QUANTITY', 'HSCODE', 'ITEM_RATE_INV', 'TOTAL_AMOUNT_INV_FC', 'FOB INR']
        categorical_features = ['FOREIGNPORT', 'FOREIGNCOUNTRY', 'INDIANPORT', 'CURRENCY']
        target = 'ITEM_RATE_INR'
        
        # Check and log missing columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            log_messages.append(f"{file_name}: Missing columns: {missing_cols}. Using available columns.")
            available_cols = [col for col in required_cols if col in data.columns]
            if not available_cols or target not in data.columns:
                raise ValueError(f"Cannot proceed: Missing critical column {target}")
            numeric_features = [col for col in numeric_features if col in data.columns]
            categorical_features = [col for col in categorical_features if col in data.columns]
        else:
            available_cols = required_cols

        # Prepare data with available columns
        X = data[numeric_features + categorical_features]
        y = data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Model pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        
        return pipeline, accuracy, X_test, y_test, y_pred
    except Exception as e:
        raise ValueError(f"Error building prediction model: {str(e)}")

def plot_export_viz_1(data, file_name):
    try:
        # Bar chart of total ITEM_RATE_INR across top 3 FOREIGNPORTs
        total_rate_by_port = data.groupby('FOREIGNPORT')['ITEM_RATE_INR'].sum().sort_values(ascending=False).head(3)
        if total_rate_by_port.empty:
            return f"Error in viz_1: No valid ports for {file_name}"
        fig, ax = plt.subplots(figsize=(10, 5))
        total_rate_by_port.plot(kind='bar', color=['#FF9999', '#66B2FF', '#99FF99'], ax=ax)
        ax.set_xlabel('Foreign Port', fontsize=10)
        ax.set_ylabel('Total Item Rate (INR)', fontsize=10)
        ax.set_title(f'Viz 1: Total Item Rate by Top 3 Ports - {file_name}', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        for i, v in enumerate(total_rate_by_port):
            ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error in viz_1: {str(e)}"

def plot_export_viz_2(data, file_name):
    try:
        # Bar chart of total QUANTITY by top 3 FOREIGNPORTs
        total_qty_by_port = data.groupby('FOREIGNPORT')['QUANTITY'].sum().sort_values(ascending=False).head(3)
        if total_qty_by_port.empty:
            return f"Error in viz_2: No valid ports for {file_name}"
        fig, ax = plt.subplots(figsize=(10, 5))
        total_qty_by_port.plot(kind='bar', color=['#FF9999', '#66B2FF', '#99FF99'], ax=ax)
        ax.set_xlabel('Foreign Port', fontsize=10)
        ax.set_ylabel('Total Quantity', fontsize=10)
        ax.set_title(f'Viz 2: Total Quantity by Top 3 Ports - {file_name}', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        for i, v in enumerate(total_qty_by_port):
            ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error in viz_2: {str(e)}"

def plot_export_viz_3(data, file_name):
    try:
        # Predictive line chart of ITEM_RATE_INR vs. 1-year prediction for top FOREIGNPORT
        if 'DATE' not in data.columns or data['ITEM_RATE_INR'].isnull().all():
            return "Error in viz_3: DATE or ITEM_RATE_INR missing"
        
        data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce')
        top_port = data.groupby('FOREIGNPORT')['ITEM_RATE_INR'].sum().idxmax()
        port_data = data[data['FOREIGNPORT'] == top_port].sort_values('DATE')
        
        if len(port_data) < 2:
            return f"Error in viz_3: Insufficient data for {top_port}"
        
        # Historical average ITEM_RATE_INR by date
        port_data = port_data.groupby(port_data['DATE'].dt.date)['ITEM_RATE_INR'].mean().reset_index()
        port_data['DATE'] = pd.to_datetime(port_data['DATE'])
        
        # Calculate historical trend and predict with moderate growth
        historical_mean = port_data['ITEM_RATE_INR'].mean()
        growth_rate = 0.02  # 2% annual growth (adjustable)
        future_days = 365  # 1 year
        future_dates = pd.date_range(start=port_data['DATE'].max() + timedelta(days=1), periods=future_days, freq='D')
        future_values = [historical_mean * (1 + growth_rate) ** (i / 365) for i in range(future_days)]
        
        # Combine historical and predicted data
        all_dates = pd.concat([port_data['DATE'], pd.Series(future_dates)])
        all_values = np.concatenate([port_data['ITEM_RATE_INR'], future_values])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(port_data['DATE'], port_data['ITEM_RATE_INR'], label='Historical Item Rate', color='blue', marker='o', linewidth=2)
        ax.plot(future_dates, future_values, label='Predicted Item Rate (1 Year)', color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Item Rate (INR)', fontsize=10)
        ax.set_title(f'Viz 3: Item Rate Trend vs. Prediction (1 Year) at {top_port} - {file_name}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Error in viz_3: {str(e)}"

# --- Streamlit App ---
st.title("Export Data Prediction Dashboard")

# Initialize log
log_messages = []

# Get all Excel files
data_folder = r"C:/Users/vardh/.vscode/export_data"
export_summary_csv = os.path.join(data_folder, "export_summary.csv")
log_file = os.path.join(data_folder, "processing_log.txt")

excel_files = glob.glob(os.path.join(data_folder, "*.xlsx")) + glob.glob(os.path.join(data_folder, "*.xls"))
log_messages.append(f"Found {len(excel_files)} Excel files: {[os.path.basename(f) for f in excel_files]}")

if not excel_files:
    st.error(f"No Excel files found in {data_folder}")
    log_messages.append(f"No Excel files found in {data_folder}")
else:
    st.write(f"Found {len(excel_files)} Excel files")

# Process files
export_summary_data = []

for file_path in excel_files:
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    log_messages.append(f"\nProcessing {file_name}...")
    st.subheader(f"Processing {file_name}...")
    
    try:
        # Read data and normalize column names
        data = pd.read_excel(file_path)
        original_columns = data.columns.tolist()
        log_messages.append(f"{file_name}: Original columns: {original_columns}")
        data = normalize_column_names(data)
        normalized_columns = data.columns.tolist()
        log_messages.append(f"{file_name}: Normalized columns: {normalized_columns}")
        
        # Verify required columns
        required_cols = ['QUANTITY', 'ITEM_RATE_INR', 'HSCODE', 'FOREIGNPORT', 'ITEM_RATE_INV', 'TOTAL_AMOUNT_INV_FC', 'FOREIGNCOUNTRY', 'INDIANPORT', 'CURRENCY']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            error_msg = f"Missing required columns: {missing_cols}"
            log_messages.append(f"{file_name}: {error_msg}")
            st.error(f"{file_name}: {error_msg}")
            continue
        
        # Validate data
        if data['ITEM_RATE_INR'].isnull().all() or data['QUANTITY'].isnull().all():
            error_msg = "Empty ITEM_RATE_INR or QUANTITY"
            log_messages.append(f"{file_name}: {error_msg}")
            st.error(f"{file_name}: {error_msg}")
            continue
        
        # Clean data
        original_len = len(data)
        data = data.dropna(subset=required_cols)
        log_messages.append(f"{file_name}: Rows after cleaning: {len(data)} (dropped {original_len - len(data)})")
        if data.empty:
            error_msg = "No valid data after cleaning"
            log_messages.append(f"{file_name}: {error_msg}")
            st.error(f"{file_name}: {error_msg}")
            continue
        
        # Build prediction model
        pipeline, accuracy, X_test, y_test, y_pred = build_prediction_model(data)
        
        # Check accuracy
        if accuracy < 0.8:
            log_messages.append(f"{file_name}: Accuracy {accuracy:.4f} below 80%, tuning model")
            # Retry with different parameters
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), ['QUANTITY', 'HSCODE', 'ITEM_RATE_INV', 'TOTAL_AMOUNT_INV_FC', 'FOB INR']),
                        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['FOREIGNPORT', 'FOREIGNCOUNTRY', 'INDIANPORT', 'CURRENCY'])
                    ]
                )),
                ('regressor', XGBRegressor(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.05,
                    random_state=42
                ))
            ])
            X_train, X_test, y_train, y_test = train_test_split(
                data[['QUANTITY', 'HSCODE', 'ITEM_RATE_INV', 'TOTAL_AMOUNT_INV_FC', 'FOB INR', 'FOREIGNPORT', 'FOREIGNCOUNTRY', 'INDIANPORT', 'CURRENCY']],
                data['ITEM_RATE_INR'],
                test_size=0.2,
                random_state=42
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = r2_score(y_test, y_pred)
        
        # Store summary
        export_summary_data.append({
            'File': file_name,
            'Avg_Predicted_ITEM_RATE_INR': y_pred.mean(),
            'Model_Accuracy': accuracy
        })
        
        # Display accuracy
        st.write(f"**Model Accuracy (R² Score)**: {accuracy:.4f}")
        
        # Generate Visualizations
        st.write("### Visualizations")
        with st.expander(f"Visualizations for {file_name}"):
            # Visualization 1: Bar Chart
            viz_1_result = plot_export_viz_1(data, file_name)
            if isinstance(viz_1_result, plt.Figure):
                st.pyplot(viz_1_result)
                log_messages.append(f"{file_name}: Visualization 1 generated")
            else:
                st.error(f"Visualization 1 failed: {viz_1_result}")
                log_messages.append(f"{file_name}: {viz_1_result}")
            
            # Visualization 2: Bar Chart
            viz_2_result = plot_export_viz_2(data, file_name)
            if isinstance(viz_2_result, plt.Figure):
                st.pyplot(viz_2_result)
                log_messages.append(f"{file_name}: Visualization 2 generated")
            else:
                st.error(f"Visualization 2 failed: {viz_2_result}")
                log_messages.append(f"{file_name}: {viz_2_result}")
            
            # Visualization 3: Line Chart
            viz_3_result = plot_export_viz_3(data, file_name)
            if isinstance(viz_3_result, plt.Figure):
                st.pyplot(viz_3_result)
                log_messages.append(f"{file_name}: Visualization 3 generated")
            else:
                st.error(f"Visualization 3 failed: {viz_3_result}")
                log_messages.append(f"{file_name}: {viz_3_result}")
        
        log_messages.append(f"{file_name}: Processed successfully")
        st.success(f"{file_name}: Processed successfully")
    
    except Exception as e:
        error_msg = f"Error - {str(e)}"
        log_messages.append(f"{file_name}: {error_msg}")
        st.error(f"{file_name}: {error_msg}")

# --- Save Summary ---
if export_summary_data:
    export_summary_df = pd.DataFrame(export_summary_data)
    export_summary_df.to_csv(export_summary_csv, index=False)
    st.write(f"Export summary saved to {export_summary_csv}")
else:
    st.error("No export files processed successfully")
    log_messages.append("No files processed successfully for summary")

# --- Save Log ---
with open(log_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_messages))
st.write(f"Log saved to {log_file}")

# --- Display Export Summary ---
st.header("Export Summary")
if export_summary_data:
    st.dataframe(export_summary_df, use_container_width=True)
else:
    st.error("No export results")