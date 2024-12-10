from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load("rf_model_97.joblib")
label_encoder = joblib.load("labelencoder.joblib")
scaler = joblib.load("scaler.joblib")


@app.route("/")
def home():
    return render_template("home_page.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        gender = request.form["gender"].strip().lower()
        education = request.form["education"].strip()
        income = int(request.form["income"])
        experience = int(request.form["experience"])
        home_ownership = request.form["home_ownership"].strip().upper()
        loan_amount = int(request.form["loan_amount"])
        loan_intent = request.form["loan_intent"].strip().upper()
        loan_interest = float(request.form["loan_interest"])
        loan_income = float(request.form["loan_income"])
        credit_history_length = float(request.form["credit_history_length"])
        credit_score = float(request.form["credit_score"])
        prev_loan_defaults = request.form["prev_loan_defaults"].strip().capitalize()


        input_data = pd.DataFrame([[
            age, gender, education, income, experience, home_ownership, loan_amount,
            loan_intent, loan_interest, loan_income, credit_history_length, credit_score, prev_loan_defaults
        ]], columns=[
            "person_age", "person_gender", "person_education", "person_income", "person_emp_exp",
            "person_home_ownership", "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file"
        ])

        categorical_columns = ["person_gender", "person_education", "person_home_ownership", "loan_intent",
                               "previous_loan_defaults_on_file"]

        input_data[categorical_columns] = label_encoder.transform(input_data[categorical_columns])



        input_data_scaled = scaler.transform(input_data)


        input_data_scaled = pd.DataFrame(input_data_scaled, columns=input_data.columns)


        prediction = model.predict(input_data_scaled)


        result = "Congratulations! Your loan is likely to be approved." if prediction[
                                                                               0] == 1 else "We are sorry. Your loan is likely to be rejected."

        return render_template("result_page.html", message=result)

    except Exception as e:
        return render_template("result_page.html", message=f"An error occurred: {e}")


if __name__ == "__main__":
    app.run(debug=True)
