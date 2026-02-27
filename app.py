# app.py
from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
predict_pipeline = PredictPipeline()

@app.route("/")
def home():
    return "Loan Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        custom_data = CustomData(**data)
        input_df = custom_data.get_data_as_dataframe()
        
        prediction, probability = predict_pipeline.predict(input_df)
        
        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability[0])  # probability of positive class
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)