import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from config.config import settings

class TextClassifier:
    def __init__(
        self,
        max_features: int = None,
    ):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=max_features)),
            ('classifier', LogisticRegression())
        ])

    def train(self, texts, labels):
        self.pipeline.fit(texts, labels)

    def predict(self, texts):
        return list(self.pipeline.predict(texts))

    def single_predict(self, text):
        return self.pipeline.predict([text])[0]

    def save(self, model_path=settings.model_save_name):
        joblib.dump(self.pipeline, model_path)

    def load(self, model_path=settings.model_save_name):
        self.pipeline = joblib.load(model_path)
