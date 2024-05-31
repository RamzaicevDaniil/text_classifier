from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.text_classifier import TextClassifier
from config.config import settings
app = FastAPI()


class TextInput(BaseModel):
    text: str


class TextBatchInput(BaseModel):
    texts: list[str]


classifier = TextClassifier()
classifier.load(settings.model_load_path)


@app.post("/predict")
def predict(input: TextInput):
    try:
        prediction = classifier.single_predict(input.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(input: TextBatchInput):
    try:
        predictions = classifier.predict(input.texts)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))