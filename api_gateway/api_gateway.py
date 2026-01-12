"""
API Gateway - Main entry point that routes requests to microservices
Lightweight service that handles routing and orchestration
"""

import os
import requests
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash, send_file
from flask_cors import CORS
import uuid
import json
from functools import lru_cache

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Microservice URLs (configure via environment variables)
PREPROCESSING_SERVICE_URL = os.environ.get('PREPROCESSING_SERVICE_URL', 'http://localhost:5001')
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5002')
DASHBOARD_SERVICE_URL = os.environ.get('DASHBOARD_SERVICE_URL', 'http://localhost:5003')

# Local storage for small files (for backward compatibility)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Session cache
session_cache = {}

# ============ FRONTEND ROUTES (unchanged for compatibility) ============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eda')
def eda():
    return render_template('PlayML.html')

@app.route('/pre')
def pre():
    return render_template('pre.html')

@app.route('/aboutus')
def aboutus():
    return render_template('about.html')

@app.route('/dash')
def dash():
    return render_template('dashboard.html')

# ============ FILE UPLOAD ROUTES ============

@app.route('/upload_preprocess', methods=['POST'])
def upload_preprocess():
    """Route to preprocessing service"""
    if 'file' not in request.files:
        flash("No file part.")
        return redirect(url_for('pre'))
    
    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(url_for('pre'))
    
    # Save file temporarily
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    
    # Forward to preprocessing service
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (file.filename, f, 'multipart/form-data')}
            response = requests.post(
                f"{PREPROCESSING_SERVICE_URL}/analyze",
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store in session for compatibility
            session['filename'] = unique_filename
            session['original_filename'] = file.filename
            session['categorical_columns'] = data.get('categorical_columns', [])
            session['column_types'] = data.get('column_types', {})
            
            return render_template('encoding_options.html',
                                 categorical_columns=data.get('categorical_columns', []),
                                 column_types=data.get('column_types', {}),
                                 filename=file.filename)
        else:
            flash(f"Preprocessing service error: {response.text}")
            return redirect(url_for('pre'))
    
    except requests.exceptions.RequestException as e:
        flash(f"Service unavailable: {str(e)}")
        return redirect(url_for('pre'))
    finally:
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    """Process encoding strategies"""
    unique_filename = session.get('filename')
    if not unique_filename:
        flash("Session expired or no file uploaded.")
        return redirect(url_for('pre'))
    
    # Collect encoding strategies
    strategies = {}
    for key, value in request.form.items():
        if key.startswith('encoding_strategy_'):
            col = key.replace('encoding_strategy_', '')
            strategies[col] = value
    
    # Forward to preprocessing service
    try:
        payload = {
            'filename': unique_filename,
            'strategies': strategies,
            'original_filename': session.get('original_filename', 'data')
        }
        
        response = requests.post(
            f"{PREPROCESSING_SERVICE_URL}/process",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Save processed file locally for download
            if 'processed_data' in data:
                processed_filename = f"processed_{session.get('original_filename', 'data')}.csv"
                processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                
                import pandas as pd
                df = pd.read_csv(data['processed_data']) if isinstance(data['processed_data'], str) else data['processed_data']
                df.to_csv(processed_path, index=False)
                
                session['processed_file'] = processed_filename
            
            return render_template('result.html',
                                 table=data.get('preview_html', ''),
                                 report=data.get('report', []),
                                 processed_file=data.get('processed_filename', ''),
                                 heatmap_image=data.get('heatmap_url', ''))
        else:
            flash(f"Processing failed: {response.text}")
            return redirect(url_for('pre'))
    
    except requests.exceptions.RequestException as e:
        flash(f"Service unavailable: {str(e)}")
        return redirect(url_for('pre'))

# ============ ML ROUTES ============

@app.route('/upload', methods=['POST'])
def upload():
    """Upload dataset for ML"""
    file = request.files.get('dataset')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Save temporarily
    uid = str(uuid.uuid4())
    filepath = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
    file.save(filepath)
    
    try:
        # Forward to ML service
        with open(filepath, 'rb') as f:
            files = {'dataset': (file.filename, f, 'multipart/form-data')}
            response = requests.post(
                f"{ML_SERVICE_URL}/upload",
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            data['uid'] = uid  # Add our local ID
            session_cache[uid] = data  # Cache response
            return jsonify(data)
        else:
            return jsonify({"error": f"ML service error: {response.text}"}), 500
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/suggest_algorithms', methods=['POST'])
def suggest_algorithms_route():
    """Forward to ML service"""
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/suggest_algorithms",
            data=request.form,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503

@app.route('/train', methods=['POST'])
def train():
    """Forward to ML service"""
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/train",
            data=request.form,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Service unavailable: {str(e)}"}), 503

# ============ DASHBOARD ROUTES ============

@app.route('/uploaddash', methods=['POST'])
def upload_file():
    """Forward to dashboard service"""
    if 'file' not in request.files:
        return render_template('dashboard.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('dashboard.html', error="No file selected")
    
    # Save temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        # Forward to dashboard service
        with open(filepath, 'rb') as f:
            files = {'file': (file.filename, f, 'multipart/form-data')}
            response = requests.post(
                f"{DASHBOARD_SERVICE_URL}/analyze",
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            return render_template('dashresult.html', **data)
        else:
            return render_template('dashboard.html', error=f"Dashboard error: {response.text}")
    
    except requests.exceptions.RequestException as e:
        return render_template('dashboard.html', error=f"Service unavailable: {str(e)}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ============ UTILITY ROUTES ============

@app.route('/download/<filename>')
def download_file(filename):
    """Serve processed files"""
    path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("File not found.")
    return redirect(url_for('pre'))

@app.route('/health')
def health_check():
    """Health check with microservice status"""
    services = {
        'api_gateway': 'healthy',
        'preprocessing': 'unknown',
        'ml': 'unknown',
        'dashboard': 'unknown'
    }
    
    # Check each service
    for name, url in [
        ('preprocessing', PREPROCESSING_SERVICE_URL),
        ('ml', ML_SERVICE_URL),
        ('dashboard', DASHBOARD_SERVICE_URL)
    ]:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                services[name] = 'healthy'
            else:
                services[name] = 'unhealthy'
        except:
            services[name] = 'unreachable'
    
    return jsonify({
        "status": "healthy",
        "services": services,
        "cache_size": len(session_cache)
    })

@app.route('/clear_cache')
def clear_cache():
    """Clear local cache"""
    session_cache.clear()
    # Also clear microservice caches
    for url in [PREPROCESSING_SERVICE_URL, ML_SERVICE_URL, DASHBOARD_SERVICE_URL]:
        try:
            requests.post(f"{url}/clear_cache")
        except:
            pass
    
    return jsonify({"message": "All caches cleared"})

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============ STARTUP ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"API Gateway starting on port {port}")
    print(f"Preprocessing Service: {PREPROCESSING_SERVICE_URL}")
    print(f"ML Service: {ML_SERVICE_URL}")
    print(f"Dashboard Service: {DASHBOARD_SERVICE_URL}")
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)