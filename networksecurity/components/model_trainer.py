print("✅ ModelTrainer file loaded from:", __file__)
import os
import sys
import joblib
import mlflow
from urllib.parse import urlparse

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# --- SAFE: guard dagshub init to avoid network calls at import time ---
try:
    import dagshub
except Exception as _e:
    logging.info("dagshub package not available; skipping dagshub.init()")
else:
    # Only initialize dagshub if explicitly enabled
    if os.getenv("ENABLE_DAGSHUB", "false").lower() in ("1", "true", "yes"):
        try:
            dagshub.init(repo_owner="dwivedivenky", repo_name="Network_security", mlflow=True)
            logging.info("✅ dagshub initialized for mlflow tracking")
        except Exception as e:
            logging.info(f"⚠️ dagshub.init() failed: {e}")
    else:
        logging.info("ENABLE_DAGSHUB not set; skipping dagshub.init()")
# --- end guarded dagshub init ---


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            logging.info("✅ ModelTrainer initialized successfully.")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):
        """
        Logs metrics and the model to MLflow (DagsHub backend)
        """
        try:
            mlflow.set_tracking_uri("https://dagshub.com/dwivedivenky/Network_security.mlflow")

            with mlflow.start_run():
                mlflow.log_metric("f1_score", classificationmetric.f1_score)
                mlflow.log_metric("precision", classificationmetric.precision_score)
                mlflow.log_metric("recall_score", classificationmetric.recall_score)

                # Save and log model
                model_filename = "model.pkl"
                joblib.dump(best_model, model_filename)
                mlflow.log_artifact(model_filename)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Trains multiple models and selects the best one based on evaluation metrics.
        """
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {"criterion": ["gini", "entropy", "log_loss"]},
                "Random Forest": {"n_estimators": [8, 16, 32, 128, 256]},
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {"learning_rate": [0.1, 0.01, 0.001], "n_estimators": [8, 16, 32, 64, 128, 256]},
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params
            )

            # Select best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"✅ Best Model Selected: {best_model_name} (score: {best_model_score})")

            # Evaluate model on train data
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            # Log with MLflow
            self.track_mlflow(best_model, classification_train_metric)

            # Evaluate on test data
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.track_mlflow(best_model, classification_test_metric)

            # Save preprocessor + model together
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            final_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=final_model)

            # also save just the model
            save_object("final_model/model.pkl", best_model)

            # Return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(f"✅ ModelTrainer Artifact Created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Loads transformed arrays, trains models, returns the training artifact.
        """
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
