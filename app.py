from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Database setup
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'loan_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ML model and scaler
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Input mappings
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
property_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}

# Database model
class LoanApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Gender = db.Column(db.String(10))
    Married = db.Column(db.String(10))
    Dependents = db.Column(db.String(10))
    Education = db.Column(db.String(20))
    Self_Employed = db.Column(db.String(10))
    ApplicantIncome = db.Column(db.Float)
    CoapplicantIncome = db.Column(db.Float)
    LoanAmount = db.Column(db.Float)
    Loan_Amount_Term = db.Column(db.Float)
    Credit_History = db.Column(db.Float)
    Property_Area = db.Column(db.String(20))
    Prediction = db.Column(db.String(20))

# Route
@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    input_values = {}

    if request.method == 'POST':
        try:
            input_values = {
                'Gender': request.form['Gender'],
                'Married': request.form['Married'],
                'Dependents': request.form['Dependents'],
                'Education': request.form['Education'],
                'Self_Employed': request.form['Self_Employed'],
                'ApplicantIncome': request.form['ApplicantIncome'],
                'CoapplicantIncome': request.form['CoapplicantIncome'],
                'LoanAmount': request.form['LoanAmount'],
                'Loan_Amount_Term': request.form['Loan_Amount_Term'],
                'Credit_History': request.form['Credit_History'],
                'Property_Area': request.form['Property_Area']
            }

            input_data = {
                'Gender': gender_map[input_values['Gender']],
                'Married': married_map[input_values['Married']],
                'Dependents': dependents_map[input_values['Dependents']],
                'Education': education_map[input_values['Education']],
                'Self_Employed': self_employed_map[input_values['Self_Employed']],
                'ApplicantIncome': float(input_values['ApplicantIncome']),
                'CoapplicantIncome': float(input_values['CoapplicantIncome']),
                'LoanAmount': float(input_values['LoanAmount']),
                'Loan_Amount_Term': float(input_values['Loan_Amount_Term']),
                'Credit_History': float(input_values['Credit_History']),
                'Property_Area': property_map[input_values['Property_Area']]
            }

            input_df = pd.DataFrame([input_data])
            numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
            input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

            result = model.predict(input_df)[0]
            prediction = 'Approved ✅' if result == 1 else 'Rejected ❌'

            # Store in DB
            loan_record = LoanApplication(
                Gender=input_values['Gender'],
                Married=input_values['Married'],
                Dependents=input_values['Dependents'],
                Education=input_values['Education'],
                Self_Employed=input_values['Self_Employed'],
                ApplicantIncome=float(input_values['ApplicantIncome']),
                CoapplicantIncome=float(input_values['CoapplicantIncome']),
                LoanAmount=float(input_values['LoanAmount']),
                Loan_Amount_Term=float(input_values['Loan_Amount_Term']),
                Credit_History=float(input_values['Credit_History']),
                Property_Area=input_values['Property_Area'],
                Prediction=prediction
            )
            db.session.add(loan_record)
            db.session.commit()

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('loan.html', prediction=prediction, values=input_values)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
