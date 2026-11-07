from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import re



app = Flask(__name__, template_folder='templates', static_folder='static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None
df = None

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_model_and_scaler():
    global model, scaler
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

def load_model_and_scaler():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

load_model_and_scaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    global model, scaler, df
    if 'csv_file' not in request.files:
        return render_template('index.html', error="No file uploaded.")
    file = request.files['csv_file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error="Invalid file. Please upload a CSV file.")
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return render_template('index.html', error="CSV file is empty.")
        if 'target' not in df.columns:
            return render_template('index.html', error="CSV must contain a 'target' column.")
        
        # Clean Data
        df.replace('?', np.nan, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(df.median(), inplace=True)
        
        # Features and Target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale Data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        
        save_model_and_scaler()
        return redirect(url_for('input_page'))
    except Exception as e:
        return render_template('index.html', error=f"Error training model: {str(e)}")

@app.route('/input')
def input_page():
    if not model or not scaler:
        return redirect(url_for('index'))
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler, df
    if not model or not scaler:
        return render_template('index.html', error="Model not trained yet. Please upload CSV first.")
    
    try:
        # Raw form values
        raw = {k: request.form.get(k, '').strip() for k in [
            'age','gender','cp','restbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
        ]}

        # Allow only digits (and one optional decimal for oldpeak); integers for age and bracketed fields
        integer_fields = ['age','gender','cp','fbs','restecg','exang','slope','ca','thal']
        for key in integer_fields:
            if not re.fullmatch(r"[-+]?\d+", raw[key] or ''):
                return render_template('input.html', error=f"Invalid {key}: only integers allowed.")

        # Numeric checks for remaining numeric fields
        for key in ['restbps','chol','thalach','oldpeak']:
            if not re.fullmatch(r"[-+]?\d+(\.\d+)?", raw[key] or ''):
                return render_template('input.html', error=f"Invalid {key}: only numbers allowed.")

        # Cast after format validation
        age = int(raw['age'])
        gender = int(raw['gender'])
        cp = int(raw['cp'])
        restbps = float(raw['restbps'])
        chol = float(raw['chol'])
        fbs = int(raw['fbs'])
        restecg = int(raw['restecg'])
        thalach = float(raw['thalach'])
        exang = int(raw['exang'])
        oldpeak = float(raw['oldpeak'])
        slope = int(raw['slope'])
        ca = int(raw['ca'])
        thal = int(raw['thal'])

        # Range restrictions for bracketed attributes
        if gender not in (0, 1):
            return render_template('input.html', error="Gender must be 0 or 1.")
        if not (0 <= cp <= 4):
            return render_template('input.html', error="Chest Pain Type must be between 0 and 4.")
        if fbs not in (0, 1):
            return render_template('input.html', error="Fasting BS must be 0 or 1.")
        if not (0 <= restecg <= 2):
            return render_template('input.html', error="Rest ECG must be between 0 and 2.")
        if exang not in (0, 1):
            return render_template('input.html', error="Exercise Angina must be 0 or 1.")
        if not (1 <= slope <= 3):
            return render_template('input.html', error="Slope must be between 1 and 3.")
        if not (0 <= ca <= 3):
            return render_template('input.html', error="Major Vessels must be between 0 and 3.")
        if thal not in (3, 6, 7):
            return render_template('input.html', error="Thal must be 3, 6, or 7.")

        # Plausibility bounds for remaining numeric fields (defensive; not from brackets)
        if not (1 <= age <= 150):
            return render_template('input.html', error="Age must be between 1 and 150.")
        if not (60 <= restbps <= 260):
            return render_template('input.html', error="Resting BP must be between 60 and 260.")
        if not (80 <= chol <= 700):
            return render_template('input.html', error="Cholesterol must be between 80 and 700.")
        if not (60 <= thalach <= 250):
            return render_template('input.html', error="Max Heart Rate must be between 60 and 250.")
        if not (0.0 <= oldpeak <= 10.0):
            return render_template('input.html', error="Oldpeak must be between 0.0 and 10.0.")

        input_data = [[age, gender, cp, restbps, chol, fbs, restecg, thalach,
                       exang, oldpeak, slope, ca, thal]]
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        # Corrected result assignment with conditional logic
        if prediction == 0:
            result = "✅ No Heart Disease"
        elif prediction == 1:
            result = "⚠ Mild Heart Disease (Low Risk)"
        elif prediction == 2:
            result = "⚠ Moderate Heart Disease (Medium Risk)"
        elif prediction == 3:
            result = "⚠ Severe Heart Disease (High Risk)"
        else:
            result = "⚠ Unknown Condition"

        # Generate Visualizations based on uploaded df
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap of Features")
        heatmap_buf = io.BytesIO()
        plt.savefig(heatmap_buf, format='png')
        heatmap_base64 = base64.b64encode(heatmap_buf.getvalue()).decode('utf-8')
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.countplot(x='target', data=df, palette='viridis')
        plt.title("Distribution of Heart Disease (Target)")
        plt.xlabel("Target (0=No Disease, 1=Disease)")
        plt.ylabel("Count")
        dist_buf = io.BytesIO()
        plt.savefig(dist_buf, format='png')
        dist_base64 = base64.b64encode(dist_buf.getvalue()).decode('utf-8')
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='coolwarm')
        plt.title("Age vs Cholesterol by Heart Disease")
        scatter_buf = io.BytesIO()
        plt.savefig(scatter_buf, format='png')
        scatter_base64 = base64.b64encode(scatter_buf.getvalue()).decode('utf-8')
        plt.close()

        importances = model.feature_importances_
        features = df.drop('target', axis=1).columns
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances[indices], y=features[indices], palette='magma')
        plt.title("Feature Importance")
        importance_buf = io.BytesIO()
        plt.savefig(importance_buf, format='png')
        importance_base64 = base64.b64encode(importance_buf.getvalue()).decode('utf-8')
        plt.close()

        return render_template('result.html', result=result, accuracy="N/A",
                              heatmap=heatmap_base64, dist=dist_base64,
                              scatter=scatter_base64, importance=importance_base64)
    except Exception as e:
        return render_template('input.html', error=f"Error predicting: {str(e)}")

@app.route('/graphs')
def graphs():
    global df, model
    if df is None or model is None:
        return render_template('index.html', error="Train the model first with a CSV.")
    
    # Generate Visualizations
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap of Features")
    heatmap_buf = io.BytesIO()
    plt.savefig(heatmap_buf, format='png')
    heatmap_base64 = base64.b64encode(heatmap_buf.getvalue()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette='viridis')
    plt.title("Distribution of Heart Disease (Target)")
    plt.xlabel("Target (0=No Disease, 1=Disease)")
    plt.ylabel("Count")
    dist_buf = io.BytesIO()
    plt.savefig(dist_buf, format='png')
    dist_base64 = base64.b64encode(dist_buf.getvalue()).decode('utf-8')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='age', y='chol', hue='target', data=df, palette='coolwarm')
    plt.title("Age vs Cholesterol by Heart Disease")
    scatter_buf = io.BytesIO()
    plt.savefig(scatter_buf, format='png')
    scatter_base64 = base64.b64encode(scatter_buf.getvalue()).decode('utf-8')
    plt.close()

    importances = model.feature_importances_
    features = df.drop('target', axis=1).columns
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=features[indices], palette='magma')
    plt.title("Feature Importance")
    importance_buf = io.BytesIO()
    plt.savefig(importance_buf, format='png')
    importance_base64 = base64.b64encode(importance_buf.getvalue()).decode('utf-8')
    plt.close()


    return render_template('graphs.html', heatmap=heatmap_base64, dist=dist_base64,
                          scatter=scatter_base64, importance=importance_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 3003)))