"""
ML Microservice - Handles all machine learning operations
Optimized with lazy loading of heavy ML libraries
"""

import os
import pandas as pd
import numpy as np
import joblib
import tempfile
import uuid
from functools import lru_cache
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_CACHE_FOLDER = 'model_cache'
DATASET_CACHE_FOLDER = 'dataset_cache'
RESULTS_FOLDER = 'results'

for folder in [MODEL_CACHE_FOLDER, DATASET_CACHE_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Cache for loaded datasets and models
dataset_cache = {}
model_cache = {}
algorithm_cache = {}

# ============ LAZY IMPORTS ============

def lazy_import_sklearn():
    """Lazy import sklearn modules"""
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        r2_score, mean_squared_error, mean_absolute_error,
        roc_auc_score, confusion_matrix, classification_report
    )
    return locals()

def lazy_import_ml_models():
    """Lazy import ML models"""
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        AdaBoostClassifier, AdaBoostRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB, BernoulliNB
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    
    # Import advanced models conditionally
    advanced_models = {}
    try:
        import xgboost as xgb
        advanced_models['xgb_classifier'] = xgb.XGBClassifier
        advanced_models['xgb_regressor'] = xgb.XGBRegressor
    except ImportError:
        pass
    
    try:
        import lightgbm as lgb
        advanced_models['lgb_classifier'] = lgb.LGBMClassifier
        advanced_models['lgb_regressor'] = lgb.LGBMRegressor
    except ImportError:
        pass
    
    try:
        import catboost as cb
        advanced_models['catboost_classifier'] = cb.CatBoostClassifier
        advanced_models['catboost_regressor'] = cb.CatBoostRegressor
    except ImportError:
        pass
    
    return advanced_models

# ============ HELPER FUNCTIONS ============

def detect_problem_type(df, target_column):
    """Detect if classification or regression"""
    if target_column not in df.columns:
        return None
    
    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(df[target_column]):
        # Check if it's actually classification (few unique values)
        unique_vals = df[target_column].nunique()
        if unique_vals <= 10 and df[target_column].dtype in [np.int64, np.int32]:
            return 'classification'
        return 'regression'
    else:
        return 'classification'

def get_recommended_algorithms(problem_type, df, target_column, time_constraint='medium'):
    """Get recommended algorithms based on problem type and dataset characteristics"""
    
    if problem_type == 'classification':
        unique_classes = df[target_column].nunique()
        
        base_algorithms = [
            {"value": "logistic", "label": "Logistic Regression", "speed": "fast"},
            {"value": "decision_tree", "label": "Decision Tree", "speed": "medium"},
            {"value": "random_forest", "label": "Random Forest", "speed": "medium"},
            {"value": "svm", "label": "SVM", "speed": "slow"},
            {"value": "knn", "label": "K-Nearest Neighbors", "speed": "medium"},
            {"value": "naive_bayes", "label": "Naive Bayes", "speed": "fast"},
        ]
        
        # Add ensemble methods for better accuracy
        ensemble_algorithms = [
            {"value": "gradient_boosting", "label": "Gradient Boosting", "speed": "medium"},
            {"value": "adaboost", "label": "AdaBoost", "speed": "medium"},
            {"value": "extra_trees", "label": "Extra Trees", "speed": "medium"},
        ]
        
        # Add neural networks for complex patterns
        neural_algorithms = [
            {"value": "mlp", "label": "Neural Network (MLP)", "speed": "slow"},
        ]
        
        # Filter by time constraint
        if time_constraint == 'fast':
            algorithms = [alg for alg in base_algorithms if alg['speed'] == 'fast']
        elif time_constraint == 'slow':
            algorithms = base_algorithms + ensemble_algorithms + neural_algorithms
        else:  # medium
            algorithms = base_algorithms + ensemble_algorithms
        
        # For binary classification, add all
        if unique_classes == 2:
            algorithms.append({"value": "xgboost", "label": "XGBoost", "speed": "medium"})
            algorithms.append({"value": "lightgbm", "label": "LightGBM", "speed": "fast"})
        
        return algorithms
    
    else:  # regression
        base_algorithms = [
            {"value": "linear_regression", "label": "Linear Regression", "speed": "fast"},
            {"value": "ridge", "label": "Ridge Regression", "speed": "fast"},
            {"value": "lasso", "label": "Lasso Regression", "speed": "fast"},
            {"value": "decision_tree", "label": "Decision Tree", "speed": "medium"},
            {"value": "random_forest", "label": "Random Forest", "speed": "medium"},
            {"value": "svr", "label": "Support Vector Regression", "speed": "slow"},
        ]
        
        ensemble_algorithms = [
            {"value": "gradient_boosting", "label": "Gradient Boosting", "speed": "medium"},
            {"value": "adaboost", "label": "AdaBoost", "speed": "medium"},
        ]
        
        if time_constraint == 'fast':
            return base_algorithms[:4]
        elif time_constraint == 'slow':
            return base_algorithms + ensemble_algorithms + [
                {"value": "xgboost", "label": "XGBoost", "speed": "medium"},
                {"value": "lightgbm", "label": "LightGBM", "speed": "fast"},
                {"value": "mlp", "label": "Neural Network (MLP)", "speed": "slow"},
            ]
        else:
            return base_algorithms + ensemble_algorithms

def preprocess_data(df, target_column):
    """Preprocess data for ML"""
    sklearn_modules = lazy_import_sklearn()
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessing pipelines
    numeric_transformer = sklearn_modules['Pipeline'](steps=[
        ('imputer', sklearn_modules['SimpleImputer'](strategy='median')),
        ('scaler', sklearn_modules['StandardScaler']())
    ])
    
    categorical_transformer = sklearn_modules['Pipeline'](steps=[
        ('imputer', sklearn_modules['SimpleImputer'](strategy='constant', fill_value='missing')),
        ('onehot', sklearn_modules['OneHotEncoder'](handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = sklearn_modules['ColumnTransformer'](
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = numeric_cols.copy()
    if categorical_cols:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = ohe.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    # Encode target if categorical
    if not pd.api.types.is_numeric_dtype(y):
        le = sklearn_modules['LabelEncoder']()
        y = le.fit_transform(y)
    
    return X_processed, y, preprocessor, feature_names

def train_model(X_train, X_test, y_train, y_test, algorithm, preprocessor=None):
    """Train a model with the specified algorithm"""
    sklearn_modules = lazy_import_sklearn()
    advanced_models = lazy_import_ml_models()
    
    model = None
    is_classification = len(np.unique(y_train)) < 10  # Simple heuristic
    
    try:
        if algorithm == 'logistic':
            model = sklearn_modules['LogisticRegression'](
                max_iter=1000, random_state=42, n_jobs=-1
            )
        elif algorithm == 'linear_regression':
            model = sklearn_modules['LinearRegression'](n_jobs=-1)
            is_classification = False
        elif algorithm == 'ridge':
            model = sklearn_modules['Ridge'](random_state=42)
            is_classification = False
        elif algorithm == 'lasso':
            model = sklearn_modules['Lasso'](random_state=42)
            is_classification = False
        elif algorithm == 'decision_tree':
            if is_classification:
                model = sklearn_modules['DecisionTreeClassifier'](
                    max_depth=10, random_state=42
                )
            else:
                model = sklearn_modules['DecisionTreeRegressor'](
                    max_depth=10, random_state=42
                )
        elif algorithm == 'random_forest':
            if is_classification:
                model = sklearn_modules['RandomForestClassifier'](
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
            else:
                model = sklearn_modules['RandomForestRegressor'](
                    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                )
        elif algorithm == 'svm' or algorithm == 'svr':
            if is_classification:
                model = sklearn_modules['SVC'](
                    probability=True, random_state=42, kernel='rbf'
                )
            else:
                model = sklearn_modules['SVR'](kernel='rbf')
        elif algorithm == 'knn':
            if is_classification:
                model = sklearn_modules['KNeighborsClassifier'](n_jobs=-1)
            else:
                model = sklearn_modules['KNeighborsRegressor'](n_jobs=-1)
        elif algorithm == 'naive_bayes':
            model = sklearn_modules['GaussianNB']()
        elif algorithm == 'gradient_boosting':
            if is_classification:
                model = sklearn_modules['GradientBoostingClassifier'](
                    n_estimators=100, random_state=42
                )
            else:
                model = sklearn_modules['GradientBoostingRegressor'](
                    n_estimators=100, random_state=42
                )
        elif algorithm == 'adaboost':
            if is_classification:
                model = sklearn_modules['AdaBoostClassifier'](random_state=42)
            else:
                model = sklearn_modules['AdaBoostRegressor'](random_state=42)
        elif algorithm == 'extra_trees':
            if is_classification:
                model = sklearn_modules['ExtraTreesClassifier'](
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                model = sklearn_modules['ExtraTreesRegressor'](
                    n_estimators=100, random_state=42, n_jobs=-1
                )
        elif algorithm == 'mlp':
            if is_classification:
                model = sklearn_modules['MLPClassifier'](
                    hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
                )
            else:
                model = sklearn_modules['MLPRegressor'](
                    hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
                )
        elif algorithm == 'xgboost' and 'xgb_classifier' in advanced_models:
            if is_classification:
                model = advanced_models['xgb_classifier'](
                    n_estimators=100, random_state=42, n_jobs=-1,
                    use_label_encoder=False, eval_metric='logloss'
                )
            else:
                model = advanced_models['xgb_regressor'](
                    n_estimators=100, random_state=42, n_jobs=-1
                )
        elif algorithm == 'lightgbm' and 'lgb_classifier' in advanced_models:
            if is_classification:
                model = advanced_models['lgb_classifier'](
                    n_estimators=100, random_state=42, n_jobs=-1
                )
            else:
                model = advanced_models['lgb_regressor'](
                    n_estimators=100, random_state=42, n_jobs=-1
                )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        if is_classification:
            accuracy = sklearn_modules['accuracy_score'](y_test, y_pred)
            precision = sklearn_modules['precision_score'](y_test, y_pred, average='weighted', zero_division=0)
            recall = sklearn_modules['recall_score'](y_test, y_pred, average='weighted', zero_division=0)
            f1 = sklearn_modules['f1_score'](y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2)
            }
            
            # ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                try:
                    roc_auc = sklearn_modules['roc_auc_score'](y_test, y_pred_proba[:, 1])
                    metrics['roc_auc'] = round(roc_auc * 100, 2)
                except:
                    pass
        else:
            r2 = sklearn_modules['r2_score'](y_test, y_pred)
            mse = sklearn_modules['mean_squared_error'](y_test, y_pred)
            mae = sklearn_modules['mean_absolute_error'](y_test, y_pred)
            
            metrics = {
                'r2_score': round(r2 * 100, 2),
                'mse': round(mse, 4),
                'rmse': round(np.sqrt(mse), 4),
                'mae': round(mae, 4)
            }
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                feature_importance = model.coef_.tolist()
            else:
                feature_importance = model.coef_[0].tolist()
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'feature_importance': feature_importance,
            'is_classification': is_classification
        }
        
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")

# ============ API ENDPOINTS ============

@app.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload and cache dataset"""
    if 'dataset' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file extension
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        return jsonify({"error": "Only CSV and Excel files are supported"}), 400
    
    # Save file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    file.save(temp_file.name)
    
    try:
        # Load dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file.name, nrows=100000)  # Limit rows for memory
        else:
            df = pd.read_excel(temp_file.name, nrows=100000)
        
        # Generate unique ID
        uid = str(uuid.uuid4())
        
        # Cache dataset
        dataset_cache[uid] = {
            'df': df,
            'path': temp_file.name,
            'filename': file.filename,
            'timestamp': pd.Timestamp.now()
        }
        
        # Save to disk for persistence
        cache_path = os.path.join(DATASET_CACHE_FOLDER, f"{uid}.joblib")
        joblib.dump(df, cache_path)
        
        # Basic info
        response = {
            "uid": uid,
            "filename": file.filename,
            "columns": list(df.columns),
            "shape": df.shape,
            "preview": df.head().to_dict(orient='records'),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
    finally:
        # Keep temp file for now
        pass

@app.route('/suggest_algorithms', methods=['POST'])
def suggest_algorithms():
    """Suggest algorithms based on dataset characteristics"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    time_constraint = request.form.get('time_constraint', 'medium')
    
    if not uid or not target:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Check cache
    cache_key = f"{uid}_{target}_{time_constraint}"
    if cache_key in algorithm_cache:
        return jsonify({"suggested_algorithms": algorithm_cache[cache_key]})
    
    # Get dataset
    if uid not in dataset_cache:
        # Try to load from disk
        cache_path = os.path.join(DATASET_CACHE_FOLDER, f"{uid}.joblib")
        if os.path.exists(cache_path):
            try:
                df = joblib.load(cache_path)
                dataset_cache[uid] = {'df': df}
            except:
                return jsonify({"error": "Dataset not found"}), 404
        else:
            return jsonify({"error": "Dataset not found"}), 404
    
    df = dataset_cache[uid]['df']
    
    if target not in df.columns:
        return jsonify({"error": "Target column not found"}), 400
    
    # Detect problem type
    problem_type = detect_problem_type(df, target)
    if not problem_type:
        return jsonify({"error": "Could not determine problem type"}), 400
    
    # Get recommended algorithms
    algorithms = get_recommended_algorithms(problem_type, df, target, time_constraint)
    
    # Cache results
    algorithm_cache[cache_key] = algorithms
    
    return jsonify({
        "suggested_algorithms": algorithms,
        "problem_type": problem_type,
        "target_info": {
            "unique_values": int(df[target].nunique()),
            "dtype": str(df[target].dtype),
            "missing_values": int(df[target].isnull().sum())
        }
    })

@app.route('/train', methods=['POST'])
def train_model_endpoint():
    """Train ML model"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')
    
    if not all([uid, target, algorithm]):
        return jsonify({"error": "Missing parameters"}), 400
    
    # Get dataset
    if uid not in dataset_cache:
        cache_path = os.path.join(DATASET_CACHE_FOLDER, f"{uid}.joblib")
        if os.path.exists(cache_path):
            try:
                df = joblib.load(cache_path)
                dataset_cache[uid] = {'df': df}
            except:
                return jsonify({"error": "Dataset not found"}), 404
        else:
            return jsonify({"error": "Dataset not found"}), 404
    
    df = dataset_cache[uid]['df']
    
    try:
        # Lazy import sklearn modules
        sklearn_modules = lazy_import_sklearn()
        
        # Preprocess data
        X_processed, y, preprocessor, feature_names = preprocess_data(df, target)
        
        # Split data
        X_train, X_test, y_train, y_test = sklearn_modules['train_test_split'](
            X_processed, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) <= 10 else None
        )
        
        # Train model
        result = train_model(X_train, X_test, y_train, y_test, algorithm)
        
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Save model
        model_path = os.path.join(MODEL_CACHE_FOLDER, f"{model_id}.joblib")
        model_data = {
            'model': result['model'],
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'metrics': result['metrics'],
            'algorithm': algorithm,
            'target': target
        }
        joblib.dump(model_data, model_path)
        
        # Cache model info
        model_cache[model_id] = {
            'path': model_path,
            'metrics': result['metrics'],
            'algorithm': algorithm
        }
        
        # Prepare response
        response = {
            "model_id": model_id,
            "algorithm": algorithm,
            "metrics": result['metrics'],
            "feature_importance_available": result['feature_importance'] is not None,
            "is_classification": result['is_classification'],
            "training_samples": len(y_train),
            "testing_samples": len(y_test)
        }
        
        # Add feature importance if available
        if result['feature_importance'] and feature_names:
            # Get top 10 features
            importance_dict = dict(zip(feature_names, result['feature_importance']))
            top_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            response['top_features'] = [
                {"feature": feat, "importance": round(imp, 4)}
                for feat, imp in top_features
            ]
        
        return jsonify(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Training error: {error_details}")
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with trained model"""
    model_id = request.form.get('model_id')
    data = request.form.get('data')  # JSON string of features
    
    if not model_id or not data:
        return jsonify({"error": "Missing parameters"}), 400
    
    # Load model
    model_path = os.path.join(MODEL_CACHE_FOLDER, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not found"}), 404
    
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        preprocessor = model_data['preprocessor']
        
        # Parse input data
        import json
        input_data = json.loads(data)
        
        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])
        
        # Preprocess input
        X_input = preprocessor.transform(df_input)
        
        # Make prediction
        prediction = model.predict(X_input)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_input)[0].tolist()
        
        response = {
            "prediction": prediction[0] if len(prediction) == 1 else prediction.tolist(),
            "probabilities": probabilities
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/model_info/<model_id>')
def get_model_info(model_id):
    """Get information about trained model"""
    if model_id in model_cache:
        return jsonify(model_cache[model_id])
    
    model_path = os.path.join(MODEL_CACHE_FOLDER, f"{model_id}.joblib")
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return jsonify({
                "algorithm": model_data.get('algorithm', 'unknown'),
                "metrics": model_data.get('metrics', {}),
                "target": model_data.get('target', 'unknown')
            })
        except:
            return jsonify({"error": "Could not load model"}), 500
    
    return jsonify({"error": "Model not found"}), 404

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "dataset_cache_size": len(dataset_cache),
        "model_cache_size": len(model_cache),
        "algorithm_cache_size": len(algorithm_cache),
        "memory_usage_mb": "OK"
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear all caches"""
    dataset_cache.clear()
    model_cache.clear()
    algorithm_cache.clear()
    
    # Clear disk cache
    for folder in [DATASET_CACHE_FOLDER, MODEL_CACHE_FOLDER]:
        for filename in os.listdir(folder):
            os.remove(os.path.join(folder, filename))
    
    return jsonify({"message": "All caches cleared"})

# ============ ADVANCED ENDPOINTS ============

@app.route('/ensemble', methods=['POST'])
def train_ensemble():
    """Train ensemble of models"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    
    if not uid or not target:
        return jsonify({"error": "Missing parameters"}), 400
    
    # This is a placeholder for ensemble training
    # In production, implement stacking/blending
    
    return jsonify({
        "message": "Ensemble training endpoint",
        "note": "Implement stacking/blending here"
    })

@app.route('/hyperparameter_tuning', methods=['POST'])
def hyperparameter_tuning():
    """Perform hyperparameter tuning"""
    uid = request.form.get('uid')
    target = request.form.get('target')
    algorithm = request.form.get('algorithm')
    
    if not all([uid, target, algorithm]):
        return jsonify({"error": "Missing parameters"}), 400
    
    # Placeholder for GridSearchCV/RandomizedSearchCV implementation
    
    return jsonify({
        "message": "Hyperparameter tuning endpoint",
        "note": "Implement GridSearchCV here"
    })

# ============ STARTUP ============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"ML Service starting on port {port}")
    print("Note: Heavy ML libraries will be loaded on-demand")
    
    # Warm up cache directories
    import shutil
    for folder in [MODEL_CACHE_FOLDER, DATASET_CACHE_FOLDER, RESULTS_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)