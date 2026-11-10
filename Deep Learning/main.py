import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import CLASS_NAMES, load_model, predict_probs
from preprocess import preprocess

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["fashion"] = load_model()
    yield
    models.clear()


app = FastAPI(title="Fashion-MNIST Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid")

    x = preprocess(img)
    if x is None:
        raise HTTPException(status_code=422, detail="Tidak ada objek terdeteksi pada gambar")

    probs = predict_probs(models["fashion"], x)
    top = int(probs.argmax())
    return {
        "prediction": CLASS_NAMES[top],
        "class_id": top,
        "confidence": float(probs.max()),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }