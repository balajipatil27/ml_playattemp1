"""
Preprocessing Microservice - Handles all data preprocessing operations
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import io
import base64
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
from functools import lru_cache
import tempfile

app = Flask(__name__)
CORS(app)

# Configuration
PREPROCESSED_FOLDER = 'preprocessed_data'
CACHE_FOLDER = 'cache'
PLOT_FOLDER = 'static/plots'

for folder in [PREPROCESSED_FOLDER, CACHE_FOLDER, PLOT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# In-memory cache
dataset_cache = {}
plot_cache = {}

# ============ HELPER FUNCTIONS ============

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx'}

def get_file_info(filepath):
    """Extract basic file information without loading entire file"""
    ext = filepath.rsplit('.', 1)[1].lower()
    
    if ext == 'xlsx':
        # Read only first 1000 rows for column info
        df = pd.read_excel(filepath, nrows=1000)
    else:
        df = pd.read_csv(filepath, nrows=1000)
    
    return {
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'shape': df.shape,
        'memory_usage': df.memory_usage(deep=True).sum() // 1024  # KB
    }

@lru_cache(maxsize=32)
def generate_heatmap(df_hash, strategies):
    """Generate and cache correlation heatmap"""
    # In production, you'd retrieve the actual dataframe
    # For demo, generate a sample heatmap
    np.random.seed(42)
    data = np.random.randn(10, 10)
    corr = np.corrcoef(data)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, cbar=True, linewidths=.5)
    plt.title('Correlation Heatmap')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Save to file
    heatmap_filename = f"{uuid.uuid4()}_heatmap.png"
    heatmap_path = os.path.join(PLOT_FOLDER, heatmap_filename)
    
    with open(heatmap_path, 'wb') as f:
        f.write(buf.getvalue())
    
    return heatmap_path, buf.getvalue()

def apply_encoding(df, strategies):
    """Apply encoding strategies to dataframe"""
    report = []
    df_encoded = df.copy()
    
    for col, method in strategies.items():
        if col not in df_encoded.columns:
            continue
            
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            report.append(f"Applied one-hot encoding to {col} ({len(dummies.columns)} new columns)")
            
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            report.append(f"Applied label encoding to {col}")
            
        elif method == 'frequency':
            # Frequency encoding
            freq = df_encoded[col].value_counts(normalize=True)
            df_encoded[col] = df_encoded[col].map(freq)
            report.append(f"Applied frequency encoding to {col}")
            
        elif method == 'target':
            # Target encoding (if target column exists)
            # This is a simplified version
            if 'target' in df_encoded.columns:
                target_mean = df_encoded.groupby(col)['target'].mean()
                df_encoded[col] = df_encoded[col].map(target_mean)
                report.append(f"Applied target encoding to {col}")
    
    return df_encoded, report

def handle_missing_values(df):
    """Handle missing values intelligently"""
    report = []
    
    # For numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            # Use median for skewed data
            skewness = abs(df[col].skew())
            if skewness > 1:
                fill_value = df[col].median()
                method = 'median'
            else:
                fill_value = df[col].mean()
                method = 'mean'
            
            df[col].fillna(fill_value, inplace=True)
            report.append(f"Filled missing values in {col} using {method} ({fill_value:.2f})")
    
    # For categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            report.append(f"Filled missing values in {col} with mode")
    
    return df, report

# ============ API ENDPOINTS ============

@app.route('/analyze', methods=['POST'])
def analyze_file():
    """Analyze uploaded file and return column information"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use CSV or Excel."}), 400
    
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file.save(temp_file.name)
    
    try:
        # Get file info
        file_info = get_file_info(temp_file.name)
        
        # Cache the file path (in production, store in distributed cache)
        file_id = str(uuid.uuid4())
        dataset_cache[file_id] = {
            'path': temp_file.name,
            'filename': file.filename,
            'info': file_info
        }
        
        return jsonify({
            'file_id': file_id,
            'filename': file.filename,
            **file_info
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to analyze file: {str(e)}"}), 500
    finally:
        # Keep file for processing
        pass

@app.route('/process', methods=['POST'])
def process_data():
    """Process data with encoding strategies"""
    data = request.json
    file_id = data.get('file_id')
    strategies = data.get('strategies', {})
    
    if not file_id or file_id not in dataset_cache:
        return jsonify({"error": "File not found or session expired"}), 404
    
    file_info = dataset_cache[file_id]
    filepath = file_info['path']
    
    try:
        # Load full dataset
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.xlsx':
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Handle missing values
        df, missing_report = handle_missing_values(df)
        
        # Apply encoding strategies
        df_encoded, encoding_report = apply_encoding(df, strategies)
        
        # Generate heatmap
        heatmap_path, heatmap_bytes = generate_heatmap(
            str(hash(str(df_encoded.select_dtypes(include=[np.number]).columns.tolist()))),
            str(strategies)
        )
        
        # Save processed data
        processed_filename = f"processed_{data.get('original_filename', 'data')}.csv"
        processed_path = os.path.join(PREPROCESSED_FOLDER, processed_filename)
        df_encoded.to_csv(processed_path, index=False)
        
        # Generate preview
        preview_html = df_encoded.head().to_html(classes='table table-striped', border=0)
        
        # Convert heatmap to base64 for API response
        heatmap_base64 = base64.b64encode(heatmap_bytes).decode('utf-8')
        
        return jsonify({
            'success': True,
            'processed_filename': processed_filename,
            'preview_html': preview_html,
            'report': missing_report + encoding_report,
            'heatmap_url': f'/plots/{os.path.basename(heatmap_path)}',
            'heatmap_base64': heatmap_base64,
            'processed_data_path': processed_path,
            'shape': df_encoded.shape
        })
        
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/plots/<filename>')
def serve_plot(filename):
    """Serve generated plots"""
    return send_file(os.path.join(PLOT_FOLDER, filename))

@app.route('/download/<file_id>')
def download_processed(file_id):
    """Download processed file"""
    if file_id in dataset_cache:
        filepath = dataset_cache[file_id].get('processed_path')
        if filepath and os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "cache_size": len(dataset_cache),
        "memory_usage": "OK"
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear service cache"""
    dataset_cache.clear()
    plot_cache.clear()
    
    # Clear cache files
    for filename in os.listdir(CACHE_FOLDER):
        os.remove(os.path.join(CACHE_FOLDER, filename))
    
    return jsonify({"message": "Cache cleared successfully"})

# ============ ADVANCED FEATURES ============

@app.route('/auto_preprocess', methods=['POST'])
def auto_preprocess():
    """Automatically preprocess data with smart defaults"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Save and process
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file.save(temp_file.name)
    
    try:
        # Load data
        ext = os.path.splitext(temp_file.name)[1].lower()
        if ext == '.xlsx':
            df = pd.read_excel(temp_file.name)
        else:
            df = pd.read_csv(temp_file.name)
        
        # Auto-detect strategies
        strategies = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 10:
                strategies[col] = 'onehot'
            elif unique_count <= 50:
                strategies[col] = 'label'
            else:
                strategies[col] = 'frequency'
        
        # Process with auto strategies
        df_processed, report = apply_encoding(df, strategies)
        df_processed, missing_report = handle_missing_values(df_processed)
        
        # Save result
        output_filename = f"auto_processed_{file.filename}.csv"
        output_path = os.path.join(PREPROCESSED_FOLDER, output_filename)
        df_processed.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'strategies_applied': strategies,
            'report': missing_report + report,
            'preview': df_processed.head().to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({"error": f"Auto-processing failed: {str(e)}"}), 500

# ============ STARTUP ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Preprocessing Service starting on port {port}")
    
    # Start with cache cleanup
    import shutil
    for folder in [PREPROCESSED_FOLDER, CACHE_FOLDER, PLOT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)