from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Charger le modèle
with open('app/randomForest_model.pkl', 'rb') as f :
    model=pickle.load(f)

@app.route('/')
def home():
    return render_template('banKApp.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Mapping
    job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
               'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
               'unemployed': 10, 'unknown': 11}
    marital_map = {'divorced': 0, 'married': 1, 'single': 2}
    education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}
    yes_no_map = {'No': 0, 'Yes': 1}
    contact_map = {'cellular': 0, 'telephone': 1, 'unknown': 2}
    month_map = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
                 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
    poutcome_map = {'failure': 0, 'nonexistent': 1, 'success': 2}

    # Récupération et transformation des données
    age = int(data['age'])
    job = job_map[data['job']]
    marital = marital_map[data['marital']]
    education = education_map[data['education']]
    default = yes_no_map[data['default']]
    balance = float(data['balance'])
    housing = yes_no_map[data['housing']]
    loan = yes_no_map[data['loan']]
    contact = contact_map[data['contact']]
    day = int(data['day'])
    month = month_map[data['month']]
    duration = int(data['duration'])
    campaign = int(data['campaign'])
    pdays = int(data['pdays'])
    previous = int(data['previous'])
    poutcome = poutcome_map[data['poutcome']]

    # Créer un array d’entrée pour le modèle
    input_data = np.array([[
        age, job, marital, education, default, balance, housing, loan,
        contact, day, month, duration, campaign, pdays, previous, poutcome
    ]])

    # Prédiction
    prediction = model.predict(input_data)[0]

    # Interprétation du résultat
    if prediction == 1:
        prediction_text = "Le client est susceptible de SOUSCRIRE à un dépôt à terme."
    else:
        prediction_text = "Le client n'est PAS susceptible de souscrire à un dépôt à terme."

    return render_template('bankk.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)