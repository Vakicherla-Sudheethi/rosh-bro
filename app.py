from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['Gender']
    #handedness = request.form['Handedness']
    age = request.form['Age']
    year_of_education = request.form['Years of education']
    socio_economic_status = request.form['Socioeconomic status']
    mini_mental_state_examination_score = request.form['Mini-Mental State Examination score']
    #clinical_dementia_rating = request.form['Clinical Dementia Rating']
    estimated_total_intracranial = request.form['Estimated total intracranial']
    normalized_whole_brain_volume = request.form['Normalized whole-brain volume']
    atlas_scaling_factor = request.form['Atlas scaling factor']
    #input_data = [[np.array()]]
    input_data = np.array([[gender, age, year_of_education, socio_economic_status, mini_mental_state_examination_score,
                          estimated_total_intracranial, normalized_whole_brain_volume, atlas_scaling_factor]])
    pred = model.predict(input_data)
    pred1 = scaler.transform(input_data)
    print(pred)
    return render_template('output.html', answer=pred1)


if __name__ == '__main__':
    app.run(port=554, debug=True)
