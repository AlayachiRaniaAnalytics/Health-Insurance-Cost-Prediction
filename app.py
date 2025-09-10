from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Charger les modèles
with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

with open('linear_model.pkl', 'rb') as file:
    linear_model = pickle.load(file)

# Dictionnaire des modèles
models = {
    'Random Forest': random_forest_model,
    'Linear Regression': linear_model,
    'Hybrid Model': None  # Placeholder pour le modèle hybride
}

# Fonction pour faire une prédiction
def make_prediction(model, features):
    features_array = np.array(features).reshape(1, -1)
    
    # Si le modèle est un modèle hybride
    if model == 'Hybrid Model':
        # Combiner les prédictions des modèles Random Forest et Linear Regression
        rf_prediction = random_forest_model.predict(features_array)[0]
        lr_prediction = linear_model.predict(features_array)[0]
        # Moyenne des prédictions
        return (rf_prediction + lr_prediction) / 2
    
    # Sinon, prédire avec le modèle sélectionné
    prediction = model.predict(features_array)
    return prediction[0]

@app.route('/')
def index():
    # Valeurs initiales du formulaire
    return render_template('index.html', prediction_text=None, age='', sex='male', bmi='', children='', smoker='yes', region='southeast', model_choice='Random Forest')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        model_choice = request.form['model_choice']

        # Encodage des données
        sex_encoded = 1 if sex == 'female' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0
        region_encoded = ['southwest', 'southeast', 'northwest', 'northeast'].index(region)

        # Préparer les features pour le modèle
        features = [age, sex_encoded, bmi, children, smoker_encoded, region_encoded]

        # Récupérer le modèle sélectionné
        model = models.get(model_choice)
        if model_choice == 'Hybrid Model':
            model = 'Hybrid Model'  # Passe un identifiant spécial pour le modèle hybride

        if not model:
            return "Erreur : Modèle choisi non valide."

        # Faire la prédiction
        prediction = make_prediction(model, features)

        # Renvoie les valeurs saisies et la prédiction
        return render_template(
            'index.html', 
            prediction_text=f"Prediction des frais médicaux: {prediction:.2f}",
            age=age, sex=sex, bmi=bmi, children=children,
            smoker=smoker, region=region, model_choice=model_choice
        )

    except Exception as e:
        return f"Erreur : {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
