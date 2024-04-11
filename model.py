class CustomEnsemble:
    def __init__(self, model_paths, weights):
        self.models = [joblib.load(path) for path in model_paths]
        self.weights = weights

    def predict_proba(self, X):
        predictions = np.array([model.predict_proba(X) for model in self.models])
        weighted_sum = np.tensordot(predictions, self.weights, axes=((0), (0)))
        return weighted_sum

    def predict(self, X):
        weighted_sum = self.predict_proba(X)
        return np.argmax(weighted_sum, axis=1)

    def evaluate(self, X, y):
        weighted_sum = self.predict_proba(X)
        final_predictions = self.predict(X)