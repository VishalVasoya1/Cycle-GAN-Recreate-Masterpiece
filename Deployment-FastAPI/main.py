
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import base64
from keras.models import load_model
from keras.utils import img_to_array, load_img
import tensorflow as tf
from PIL import Image
import io
from tensorflow_addons.layers import InstanceNormalization

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # function -> runs before every request
    allow_origins=["*"], # what domain should we able to talk to our api
    allow_credentials=True, # 
    allow_methods=["*"], # not only allow spicific domain but also we allow specific http method -> if public api -> user get data -> we not allow user to make put request 
    allow_headers=["*"], # spcific header
)

app.mount("/static", StaticFiles(directory="Deployment-FastAPI/static"), name="static")

templates = Jinja2Templates(directory="Deployment-FastAPI/templates")

# Load the TensorFlow model

model_1 = load_model('Models/g_model_AtoB_001000.h5')
model_2 = load_model('Models/g_model_BtoA_001000.h5')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/")
# def home():
#     return {"Status":"running fastapi succesfully"}


def preprocessing(file):
    # input_image = Image.open().convert('RGB')
    input_image = load_img(io.BytesIO(file), target_size=(256,256))
    input_image = img_to_array(input_image)
    input_image = (input_image - 127.5) / 127.5
    input_image = tf.expand_dims(input_image, 0)
    return input_image

@app.post("/generate_images")
async def upload_file(file: UploadFile = File(...)):

    content = await file.read()
    input_image = preprocessing(content)
    generated_image_1 = model_1(input_image)[0]
    generated_image_1 = tf.image.convert_image_dtype(generated_image_1, tf.uint8)
    generated_image_1 = Image.fromarray(generated_image_1.numpy())

    generated_image_2 = model_2(input_image)[0]
    generated_image_2 = tf.image.convert_image_dtype(generated_image_2, tf.uint8)
    generated_image_2 = Image.fromarray(generated_image_2.numpy())

    buffered_1 = io.BytesIO()
    generated_image_1.save(buffered_1, format="JPEG")
    img_str_1 = base64.b64encode(buffered_1.getvalue()).decode("ascii")

    buffered_2 = io.BytesIO()
    generated_image_2.save(buffered_2, format="JPEG")
    img_str_2 = base64.b64encode(buffered_2.getvalue()).decode("ascii")


    return { "image_1" : img_str_1 , "image_2" : img_str_2 }
