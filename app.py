# A very simple Flask Hello World app for you to get started with...
import time
import pandas as pd
import os
from flask import Flask, jsonify, request, render_template,make_response
import pickle

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

app = Flask(__name__)
def format_server_time():
    server_time = time.localtime()
    return time.strftime("%I:%M:%S %p", server_time)

MODEL_DIR = os.path.join(
    app.root_path,
    "static",
    "model",
)
OTHERS_DIR = os.path.join(
    app.root_path,
    "static",
    "others",
)

# Models
with open(os.path.join(MODEL_DIR,"model_nonprecious_lightgbm_tuned.pkl"), "rb") as f:
    precious_model = pickle.load(f)
with open(os.path.join(MODEL_DIR,"model_nonprecious_lightgbm_tuned.pkl"), "rb") as f:
    non_precious_model = pickle.load(f)

# Encoder and Scaler
with open(os.path.join(OTHERS_DIR,"labelencoder.pkl"), 'rb') as file:
    labelencoder = pickle.load(file)
with open(os.path.join(OTHERS_DIR,"pscaler.pkl"), 'rb') as file:
        pscaler = pickle.load(file)
with open(os.path.join(OTHERS_DIR,"npscaler.pkl"), 'rb') as file:
    npscaler = pickle.load(file)

def preprocessing(is_precious, X):
    if is_precious:
        X_scaled = pscaler.transform(X)
        results_precious = precious_model.predict(X_scaled)
        return results_precious
    else:
        X_scaled = npscaler.transform(X)
        results_non_precious = non_precious_model.predict(X_scaled)
        return results_non_precious


@app.route('/',methods=['GET', 'POST'])
def index():
    template = render_template('index.html')
    response = make_response(template)
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route("/predict", methods=["POST"])
def predict():
    feature_cols = ['form_encoded', 'Acid', 'Alcohol', 'Ketone', 'Aldehyde', 'Furan', 
                'Sugar', 'Phenol', 'Ester', 'Ether', 'Cat loading (%wt)', 
                'Temp (c)', 'time (h)', 'Pressure (bar)']
    data = request.json
    app.logger.info(f'output :{data}')

    # Oil components
    is_precious = data['is_precious']
    form = data['form']
    Acid = data['Acid']
    Alcohol = data['Alcohol']
    Ketone = data['Ketone']
    Aldehyde = data['Aldehyde']
    Furan = data['Furan']
    Sugar = data['Sugar']
    Phenol = data['Phenol']
    Ester = data['Ester']
    Ether = data['Ether']

    # Operation condition
    Cl = data['Cl']
    Temp = data['Temp']
    Time = data['Time']
    Pressure = data['Pressure']

    if(Acid == '0' and Alcohol == '0' and Ketone == '0' and Aldehyde == '0' and Furan == '0' and Sugar == '0' and
       Phenol == '0' and Ester == '0' and Ether == '0' and Cl == '0' and Temp == '0' and Time == '0' and Pressure == '0'):
        return jsonify({"error": "Invalid input"}), 400

    clean_data = {}
    for col in feature_cols:
        try:
            if col == 'form_encoded':
                clean_data[col] = float(labelencoder.transform(form))
            else:
                clean_data[col] = float(data[col])
        except (KeyError, ValueError, TypeError):
            return jsonify({"error": f"Invalid or missing value for {col}"}), 400

    entry = pd.DataFrame([clean_data]).to_numpy()
    app.logger.info("Stucked at predict")
    pred = preprocessing(is_precious,entry)
    app.logger.info("Passed at predict")
    res = {'prediction':pred}

    return jsonify(res)

@app.route('/Procedure',methods=['GET'])
def algorithm():
    # 1
    template = render_template('procedure.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route('/Datasets',methods=['GET'])
def dataset():
    # 1
    template = render_template('dataset.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response

@app.route('/Researcher',methods=['GET'])
def researcher():
    # 1
    template = render_template('researcher.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response
@app.route('/About',methods=['GET'])
def about():

    # 1
    template = render_template('about.html')
    # 2
    response = make_response(template)
    # 3
    # response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=600'
    return response
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


