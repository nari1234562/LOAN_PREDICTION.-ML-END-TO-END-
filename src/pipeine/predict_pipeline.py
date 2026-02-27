# predict_pipeline.py
class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = "artifacts/model.pkl"
            self.preprocessor_path = "artifacts/preprocessor.pkl"
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df):
        try:
            input_arr = self.preprocessor.transform(df)
            predictions = self.model.predict(input_arr)
            
            # Get probability of positive class (1)
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(input_arr)[:, 1]  # prob of class 1
            else:
                probabilities = [None] * len(predictions)  # fallback

            return predictions, probabilities
        except Exception as e:
            raise CustomException(e, sys)