# app.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app=application
predict_pipeline = PredictPipeline()  # Load model + preprocessor once at startup
THRESHOLD = 0.6  # Threshold for classifying rejection





@app.route("/", methods=["GET"])
def home():
    """
    Render HTML web form for user input
    """
    return render_template("index.html")


@app.route("/predict_web", methods=["POST"])
def predict_web():
    """
    Endpoint used by the web form
    """
    try:
        
        data = {
            "age": int(request.form.get("age")),
            "gender": request.form.get("gender"),
            "education_level": request.form.get("education_level"),
            "annual_income": float(request.form.get("annual_income")),
            "employment_experience_years": float(request.form.get("employment_experience_years")),
            "home_ownership_status": request.form.get("home_ownership_status"),
            "loan_amount": float(request.form.get("loan_amount")),
            "loan_purpose": request.form.get("loan_purpose"),
            "interest_rate": float(request.form.get("interest_rate")),
            "loan_to_income_ratio": float(request.form.get("loan_to_income_ratio")),
            "credit_history_length_years": float(request.form.get("credit_history_length_years")),
            "credit_score": float(request.form.get("credit_score")),
            "prior_default_flag": request.form.get("prior_default_flag")
        }

      
        custom_data = CustomData(**data)
        input_df = custom_data.get_data_as_dataframe()

        
        prediction, probability = predict_pipeline.predict(input_df)
        prob = float(probability[0]) if probability[0] is not None else None
        status = "Loan Rejected" if prob >= THRESHOLD else "Loan Approved" if prob is not None else "Unknown"

        return render_template(
            "index.html",
            prediction=int(prediction[0]),
            status=status,
            probability=prob,
            form_data=data
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)