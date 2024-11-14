# main.py
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(title="MNIST 손글씨 예측 서비스")

# 정적 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 모델 로드 함수
def load_model():
    try:
        model = tf.keras.models.load_model('./mnist_model_20241111_160506.keras')  # TensorFlow 모델 로드
        print("모델이 성공적으로 로드되었습니다.")
        return model
    except Exception as e:
        print(f"모델 로드 중 에러 발생: {str(e)}")
        raise Exception("모델을 로드할 수 없습니다.")


# MNIST 이미지 전처리 함수
def preprocess_image(image: Image.Image):
    # 흑백으로 변환
    image = image.convert("L")

    # MNIST 크기인 28x28로 리사이즈
    image = image.resize((28, 28))

    # numpy 배열로 변환
    image_array = np.array(image)

    # 정규화 (0-1 범위로)
    image_array = image_array / 255.0

    # 모델 입력 shape로 변환 (1, 28, 28, 1)
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array


# MNIST 클래스 레이블 (0-9)
CLASS_LABELS = [str(i) for i in range(10)]

# 시작시 모델 로드
model = load_model()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 파일 검증
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

        # 이미지 읽기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 이미지 전처리
        processed_image = preprocess_image(image)

        # 예측
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction[0])
        probability = float(np.max(prediction[0]))

        return {
            "prediction": str(predicted_digit),
            "probability": probability,
            "probabilities": {
                label: float(prob) 
                for label, prob in zip(CLASS_LABELS, prediction[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
