"""
Dashboard Microservice - Handles all visualization and dashboard operations
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import io
import base64
import json
import tempfile
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Configuration
PLOT_FOLDER = 'dashboard_plots'
CACHE_FOLDER = 'dashboard_cache'
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# Cache for generated plots
plot_cache = {}

# ============ VISUALIZATION FUNCTIONS ============

def generate_basic_plots(df):
    """Generate basic visualization plots"""
    plots = []
    
    # 1. Distribution plot for first numerical column
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        col = numerical_cols[0]
        df[col].dropna().hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {col}', fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        
        plot_id = f"{uuid.uuid4()}_dist.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_id)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        plots.append({
            'type': 'distribution',
            'title': f'Distribution of {col}',
            'path': plot_path,
            'col': col
        })
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap', fontsize=14)
        
        plot_id = f"{uuid.uuid4()}_corr.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_id)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        plots.append({
            'type': 'correlation',
            'title': 'Correlation Heatmap',
            'path': plot_path
        })
    
    # 3. Scatter plot for first two numerical columns
    if len(numerical_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        col1, col2 = numerical_cols[0], numerical_cols[1]
        ax.scatter(df[col1], df[col2], alpha=0.6, s=30)
        ax.set_title(f'{col1} vs {col2}', fontsize=14)
        ax.set_xlabel(col1, fontsize=12)
        ax.set_ylabel(col2, fontsize=12)
        
        plot_id = f"{uuid.uuid4()}_scatter.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_id)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        plots.append({
            'type': 'scatter',
            'title': f'{col1} vs {col2}',
            'path': plot_path,
            'cols': [col1, col2]
        })
    
    # 4. Bar plot for first categorical column
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        col = categorical_cols[0]
        value_counts = df[col].value_counts().head(10)  # Top 10
        value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'Top 10 Values in {col}', fontsize=14)
        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        plot_id = f"{uuid.uuid4()}_bar.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_id)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        plots.append({
            'type': 'bar',
            'title': f'Top 10 Values in {col}',
            'path': plot_path,
            'col': col
        })
    
    # 5. Box plot
    if len(numerical_cols) > 0 and len(categorical_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        num_col = numerical_cols[0]
        cat_col = categorical_cols[0]
        
        # Limit to top 5 categories for readability
        top_categories = df[cat_col].value_counts().head(5).index
        df_filtered = df[df[cat_col].isin(top_categories)]
        
        df_filtered.boxplot(column=num_col, by=cat_col, ax=ax, grid=False)
        ax.set_title(f'{num_col} by {cat_col}', fontsize=14)
        ax.set_xlabel(cat_col, fontsize=12)
        ax.set_ylabel(num_col, fontsize=12)
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45, ha='right')
        
        plot_id = f"{uuid.uuid4()}_box.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_id)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        plots.append({
            'type': 'box',
            'title': f'{num_col} by {cat_col}',
            'path': plot_path,
            'cols': [num_col, cat_col]
        })
    
    return plots

def generate_interactive_plots(df):
    """Generate interactive Plotly plots"""
    interactive_plots = []
    
    # 1. Interactive correlation matrix
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().round(2)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Interactive Correlation Matrix',
            width=800,
            height=600
        )
        
        plot_html = pio.to_html(fig, full_html=False)
        interactive_plots.append({
            'type': 'interactive_correlation',
            'title': 'Interactive Correlation Matrix',
            'html': plot_html
        })
    
    # 2. Interactive scatter matrix
    if len(numerical_cols) >= 3:
        # Take first 3 numerical columns for scatter matrix
        cols_to_plot = numerical_cols[:3]
        
        fig = go.Figure(data=go.Splom(
            dimensions=[dict(label=col, values=df[col]) for col in cols_to_plot],
            showupperhalf=False,
            diagonal_visible=False,
            marker=dict(
                size=5,
                colorscale='Blues',
                showscale=False,
                line_color='white',
                line_width=0.5
            )
        ))
        
        fig.update_layout(
            title='Scatter Matrix',
            width=800,
            height=600,
            dragmode='select',
            hovermode='closest'
        )
        
        plot_html = pio.to_html(fig, full_html=False)
        interactive_plots.append({
            'type': 'scatter_matrix',
            'title': 'Scatter Matrix',
            'html': plot_html
        })
    
    return interactive_plots

def calculate_kpis(df):
    """Calculate Key Performance Indicators"""
    kpis = []
    
    # Basic KPIs
    kpis.append({
        'label': 'Total Rows',
        'value': f"{len(df):,}",
        'description': 'Number of records in dataset'
    })
    
    kpis.append({
        'label': 'Total Columns',
        'value': f"{len(df.columns):,}",
        'description': 'Number of features in dataset'
    })
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    kpis.append({
        'label': 'Memory Usage',
        'value': f"{memory_mb:.2f} MB",
        'description': 'Approximate memory footprint'
    })
    
    # Missing values
    missing_total = df.isnull().sum().sum()
    missing_percentage = (missing_total / (len(df) * len(df.columns))) * 100
    kpis.append({
        'label': 'Missing Values',
        'value': f"{missing_total:,} ({missing_percentage:.1f}%)",
        'description': 'Total missing values in dataset'
    })
    
    # Data types
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
    kpis.append({
        'label': 'Numeric Columns',
        'value': f"{numeric_cols}",
        'description': 'Number of numerical features'
    })
    
    kpis.append({
        'label': 'Categorical Columns',
        'value': f"{categorical_cols}",
        'description': 'Number of categorical features'
    })
    
    return kpis

def generate_statistics(df):
    """Generate detailed statistics"""
    stats = {}
    
    # Basic statistics
    stats['basic'] = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    # Descriptive statistics for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats['numerical'] = df[numerical_cols].describe().round(2).to_dict()
    else:
        stats['numerical'] = {}
    
    # Descriptive statistics for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            cat_stats[col] = {
                'unique_values': int(df[col].nunique()),
                'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'missing_values': int(df[col].isnull().sum())
            }
        stats['categorical'] = cat_stats
    else:
        stats['categorical'] = {}
    
    # Correlation matrix (top 10 correlations)
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs().unstack().sort_values(ascending=False)
        top_correlations = []
        seen_pairs = set()
        
        for idx, value in corr_matrix.items():
            if idx[0] != idx[1] and idx not in seen_pairs and (idx[1], idx[0]) not in seen_pairs:
                if len(top_correlations) < 10:  # Top 10 correlations
                    top_correlations.append({
                        'feature1': idx[0],
                        'feature2': idx[1],
                        'correlation': round(value, 3)
                    })
                    seen_pairs.add(idx)
                else:
                    break
        
        stats['top_correlations'] = top_correlations
    
    return stats

# ============ API ENDPOINTS ============

@app.route('/analyze', methods=['POST'])
def analyze_dashboard():
    """Analyze uploaded file for dashboard"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file type
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400
    
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file.save(temp_file.name)
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file.name, nrows=100000)  # Limit rows
        else:
            df = pd.read_excel(temp_file.name, nrows=100000)
        
        # Generate dashboard components
        plots = generate_basic_plots(df)
        interactive_plots = generate_interactive_plots(df)
        kpis = calculate_kpis(df)
        stats = generate_statistics(df)
        
        # Convert plots to base64 for API response
        plot_data = []
        for plot in plots:
            with open(plot['path'], 'rb') as f:
                plot_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            plot_data.append({
                'type': plot['type'],
                'title': plot['title'],
                'data': plot_base64,
                'details': {k: v for k, v in plot.items() if k not in ['path', 'data']}
            })
        
        # Prepare response
        response = {
            'filename': file.filename,
            'shape': df.shape,
            'plots': plot_data,
            'interactive_plots': interactive_plots,
            'kpis': kpis,
            'statistics': stats,
            'preview': df.head(10).to_dict(orient='records'),
            'columns': list(df.columns)
        }
        
        # Cache the analysis
        analysis_id = str(uuid.uuid4())
        cache_path = os.path.join(CACHE_FOLDER, f"{analysis_id}.json")
        with open(cache_path, 'w') as f:
            json.dump(response, f)
        
        response['analysis_id'] = analysis_id
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

@app.route('/plot/<plot_id>')
def serve_plot(plot_id):
    """Serve generated plot image"""
    plot_path = os.path.join(PLOT_FOLDER, plot_id)
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({"error": "Plot not found"}), 404

@app.route('/custom_plot', methods=['POST'])
def create_custom_plot():
    """Create custom visualization based on parameters"""
    data = request.json
    plot_type = data.get('type')
    columns = data.get('columns', [])
    analysis_id = data.get('analysis_id')
    
    if not plot_type or not columns:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Load cached data if available
    df = None
    if analysis_id:
        cache_path = os.path.join(CACHE_FOLDER, f"{analysis_id}.json")
        if os.path.exists(cache_path):
            # In production, you'd reload the actual dataframe
            # For now, we'll create a sample response
            pass
    
    # Generate custom plot based on type
    plot_id = f"{uuid.uuid4()}_custom.png"
    plot_path = os.path.join(PLOT_FOLDER, plot_id)
    
    try:
        if plot_type == 'histogram' and len(columns) >= 1:
            # Generate histogram
            plt.figure(figsize=(10, 6))
            # In production, use actual data
            # For demo, create sample data
            sample_data = np.random.normal(0, 1, 1000)
            plt.hist(sample_data, bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Histogram of {columns[0]}', fontsize=14)
            plt.xlabel(columns[0], fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
        elif plot_type == 'scatter' and len(columns) >= 2:
            # Generate scatter plot
            plt.figure(figsize=(10, 6))
            # Sample data
            x = np.random.normal(0, 1, 100)
            y = x + np.random.normal(0, 0.5, 100)
            plt.scatter(x, y, alpha=0.6)
            plt.title(f'{columns[0]} vs {columns[1]}', fontsize=14)
            plt.xlabel(columns[0], fontsize=12)
            plt.ylabel(columns[1], fontsize=12)
            
        elif plot_type == 'bar' and len(columns) >= 1:
            # Generate bar plot
            plt.figure(figsize=(10, 6))
            categories = ['A', 'B', 'C', 'D', 'E']
            values = np.random.randint(10, 100, len(categories))
            plt.bar(categories, values, alpha=0.7, edgecolor='black')
            plt.title(f'Bar Plot of {columns[0]}', fontsize=14)
            plt.xlabel(columns[0], fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        # Convert to base64
        with open(plot_path, 'rb') as f:
            plot_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'plot_id': plot_id,
            'type': plot_type,
            'columns': columns,
            'data': plot_base64
        })
        
    except Exception as e:
        return jsonify({"error": f"Plot generation failed: {str(e)}"}), 500

@app.route('/export/<analysis_id>/<format>')
def export_analysis(analysis_id, format):
    """Export analysis results"""
    cache_path = os.path.join(CACHE_FOLDER, f"{analysis_id}.json")
    if not os.path.exists(cache_path):
        return jsonify({"error": "Analysis not found"}), 404
    
    with open(cache_path, 'r') as f:
        analysis_data = json.load(f)
    
    if format == 'json':
        return jsonify(analysis_data)
    elif format == 'html':
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .plot {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
                .kpi {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                .stat {{ margin: 10px 0; padding: 10px; background: #e9ecef; }}
            </style>
        </head>
        <body>
            <h1>Dashboard Analysis Report</h1>
            <h2>File: {analysis_data.get('filename', 'Unknown')}</h2>
            <p>Shape: {analysis_data.get('shape', [0, 0])}</p>
            
            <h3>Key Performance Indicators</h3>
            <div>
        """
        
        for kpi in analysis_data.get('kpis', []):
            html_content += f"""
                <div class="kpi">
                    <strong>{kpi['label']}:</strong> {kpi['value']}<br>
                    <small>{kpi.get('description', '')}</small>
                </div>
            """
        
        html_content += """
            </div>
            
            <h3>Visualizations</h3>
        """
        
        for plot in analysis_data.get('plots', []):
            html_content += f"""
                <div class="plot">
                    <h4>{plot['title']}</h4>
                    <img src="data:image/png;base64,{plot['data']}" alt="{plot['title']}">
                </div>
            """
        
        html_content += """
            </body>
            </html>
        """
        
        return html_content, 200, {'Content-Type': 'text/html'}
    
    return jsonify({"error": "Unsupported format"}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "plot_cache_size": len(plot_cache),
        "storage_available": "OK"
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear dashboard cache"""
    plot_cache.clear()
    
    # Clear plot files
    for filename in os.listdir(PLOT_FOLDER):
        os.remove(os.path.join(PLOT_FOLDER, filename))
    
    # Clear analysis cache
    for filename in os.listdir(CACHE_FOLDER):
        os.remove(os.path.join(CACHE_FOLDER, filename))
    
    return jsonify({"message": "Dashboard cache cleared"})

# ============ ADVANCED VISUALIZATION ============

@app.route('/advanced_plot', methods=['POST'])
def advanced_plot():
    """Create advanced visualization with more options"""
    data = request.json
    
    plot_type = data.get('type', 'histogram')
    columns = data.get('columns', [])
    options = data.get('options', {})
    
    # Generate advanced plot
    plot_id = f"{uuid.uuid4()}_advanced.png"
    plot_path = os.path.join(PLOT_FOLDER, plot_id)
    
    try:
        plt.figure(figsize=options.get('figsize', (12, 8)))
        
        if plot_type == 'pairplot':
            # Simplified pairplot
            n_cols = min(len(columns), 4)
            fig, axes = plt.subplots(n_cols, n_cols, figsize=(15, 15))
            
            for i in range(n_cols):
                for j in range(n_cols):
                    if i == j:
                        axes[i, j].hist(np.random.normal(0, 1, 100), bins=20, alpha=0.7)
                        axes[i, j].set_title(columns[i])
                    else:
                        x = np.random.normal(0, 1, 100)
                        y = x + np.random.normal(0, 0.5, 100)
                        axes[i, j].scatter(x, y, alpha=0.5, s=20)
                    
                    if j == 0:
                        axes[i, j].set_ylabel(columns[i])
                    if i == n_cols - 1:
                        axes[i, j].set_xlabel(columns[j])
            
            plt.suptitle('Pair Plot', fontsize=16)
            
        elif plot_type == 'violin':
            # Violin plot
            data_to_plot = [np.random.normal(0, 1, 100) for _ in range(len(columns))]
            plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
            plt.xticks(range(1, len(columns) + 1), columns)
            plt.title('Violin Plot', fontsize=14)
            plt.ylabel('Value')
            
        elif plot_type == 'heatmap_custom':
            # Custom heatmap
            matrix_size = options.get('size', 8)
            data = np.random.rand(matrix_size, matrix_size)
            plt.imshow(data, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Custom Heatmap', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        
        # Convert to base64
        with open(plot_path, 'rb') as f:
            plot_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return jsonify({
            'plot_id': plot_id,
            'type': plot_type,
            'data': plot_base64,
            'options': options
        })
        
    except Exception as e:
        return jsonify({"error": f"Advanced plot failed: {str(e)}"}), 500

# ============ STARTUP ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Dashboard Service starting on port {port}")
    
    # Set style for plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Clean up old plots
    import shutil
    if os.path.exists(PLOT_FOLDER):
        shutil.rmtree(PLOT_FOLDER)
    os.makedirs(PLOT_FOLDER)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)