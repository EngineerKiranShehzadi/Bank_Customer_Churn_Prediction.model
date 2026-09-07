# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# app = Flask(__name__)

# # Load the pre-trained model
# model = joblib.load('gradient_boosting_model.pkl')

# # Initialize the scaler (used for scaling numerical features)
# scaler = StandardScaler()

# # Route to display the home page (input form)
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route to handle the form submission and make prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the form
#     age = int(request.form['age'])
#     credit_score = int(request.form['credit_score'])
#     balance = float(request.form['balance'])
#     products_number = int(request.form['products_number'])
#     active_member = int(request.form['active_member'])
#     estimated_salary = float(request.form['estimated_salary'])
    
#     # Get the categorical input data
#     country = request.form['country']
#     gender = request.form['gender']
#     tenure = int(request.form['tenure'])
#     credit_card = int(request.form['credit_card'])
    
#     # One-hot encode country and gender
#     country_france = 1 if country == 'France' else 0
#     country_spain = 1 if country == 'Spain' else 0
#     gender_male = 1 if gender == 'Male' else 0
#     gender_female = 1 if gender == 'Female' else 0
    
#     # Prepare the input data in the format expected by the model (11 features)
#     input_data = pd.DataFrame([[age, credit_score, balance, products_number, credit_card, active_member, 
#                                 estimated_salary, tenure, country_france, country_spain, gender_male]],
#                               columns=['age', 'credit_score', 'balance', 'products_number', 'credit_card', 
#                                        'active_member', 'estimated_salary', 'tenure', 'country_France', 'country_Spain', 
#                                        'gender_Male'])  # Corrected column names for 11 features

#     # Print to debug and check if there are extra features
#     print(input_data)  # Check the input data shape and columns

#     # Scale the input data using the same scaler used during model training
#     input_data_scaled = scaler.fit_transform(input_data)

#     # Make a prediction using the loaded model
#     prediction = model.predict(input_data_scaled)
    
#     # Return the result
#     if prediction == 1:
#         result = "Churn"
#     else:
#         result = "No Churn"
    
#     return render_template('index.html', prediction_text=f"Prediction: {result}")

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model + scaler + column list + evaluation metadata.
# The decision threshold is tuned on out-of-fold predictions during
# training (see bank_customer_churn_model.py) and persisted here so
# serving matches evaluation instead of silently using 0.5.
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")
train_columns = joblib.load("train_columns.pkl")
churn_metadata = joblib.load("churn_metadata.pkl")
DECISION_THRESHOLD = churn_metadata["threshold"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Read user inputs
    age = int(request.form["age"])
    credit_score = int(request.form["credit_score"])
    balance = float(request.form["balance"])
    products_number = int(request.form["products_number"])
    credit_card = int(request.form["credit_card"])
    active_member = int(request.form["active_member"])
    estimated_salary = float(request.form["estimated_salary"])
    tenure = int(request.form["tenure"])
    country = request.form["country"]
    gender = request.form["gender"]

    # 2) Put into DataFrame EXACTLY like raw training data
    input_df = pd.DataFrame([{
        "age": age,
        "credit_score": credit_score,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary,
        "tenure": tenure,
        "country": country,
        "gender": gender
    }])

    # 3) One-hot encode same as training
    input_df = pd.get_dummies(input_df, drop_first=True)

    # 4) Align columns to training columns
    input_df = input_df.reindex(columns=train_columns, fill_value=0)

    # 5) Scale using training scaler (transform only!)
    input_scaled = scaler.transform(input_df)

    # 6) Predict using the tuned decision threshold, not the model's
    #    default 0.5 cutoff
    churn_probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(churn_probability >= DECISION_THRESHOLD)

    result = "Churn" if prediction == 1 else "No Churn"
    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result} (probability: {churn_probability:.2%})",
    )

if __name__ == "__main__":
    app.run(debug=True)

