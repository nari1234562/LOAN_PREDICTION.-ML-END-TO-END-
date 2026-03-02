def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        threshold = 0.6

        for model_name, model in models.items():

            print("\n====")
            print(f"MODEL: {model_name}")
            print("======")

            model_params = param.get(model_name, {})

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=model_params,
                cv=3,
                scoring="f1",
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Predictions
            if hasattr(best_model, "predict_proba"):
                y_train_prob = best_model.predict_proba(X_train)[:, 1]
                y_test_prob = best_model.predict_proba(X_test)[:, 1]

                y_train_pred = (y_train_prob >= threshold).astype(int)
                y_test_pred = (y_test_prob >= threshold).astype(int)

                train_auc = roc_auc_score(y_train, y_train_prob)
                test_auc = roc_auc_score(y_test, y_test_prob)
            else:
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                train_auc = None
                test_auc = None

            # Metrics
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)

            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            cm = confusion_matrix(y_test, y_test_pred)

            # ---------------- MLflow Logging ----------------
            with mlflow.start_run(run_name=model_name):

                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("stage", "training")

                mlflow.log_params(grid_search.best_params_)

                mlflow.log_metric("train_precision", train_precision)
                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("train_f1", train_f1)

                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1", test_f1)

                if train_auc is not None:
                    mlflow.log_metric("train_auc", train_auc)
                    mlflow.log_metric("test_auc", test_auc)

                mlflow.sklearn.log_model(best_model, artifact_path="model")

            print("\nTrain Metrics:")
            print(f"Precision: {train_precision:.4f}  Recall: {train_recall:.4f}  F1: {train_f1:.4f}")
            if train_auc is not None:
                print(f"ROC-AUC: {train_auc:.4f}")

            print("\nTest Metrics:")
            print(f"Precision: {test_precision:.4f}  Recall: {test_recall:.4f}  F1: {test_f1:.4f}")
            if test_auc is not None:
                print(f"ROC-AUC: {test_auc:.4f}")

            print("\nConfusion Matrix:")
            print(cm)

            report[model_name] = {
                "model": best_model,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "test_auc": test_auc
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)