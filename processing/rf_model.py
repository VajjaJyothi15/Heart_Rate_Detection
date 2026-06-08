import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


class RFHeartRateModel:
    """
    Random Forest regression layer for BPM refinement
    """

    def __init__(self, model_path="models/rf_model.pkl"):

        self.model_path = model_path

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = self._train_dummy_model()
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, model_path)

    def _train_dummy_model(self):

        X = np.random.rand(2000, 5)

        y = (
            X[:, 0] * 30 +
            X[:, 1] * 20 +
            X[:, 2] * 25 +
            60
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        return model

    def predict(self, features):

        features = np.array(features).reshape(1, -1)

        return float(self.model.predict(features)[0])