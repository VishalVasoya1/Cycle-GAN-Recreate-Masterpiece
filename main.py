from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import base64
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import io
from tensorflow_addons.layers import InstanceNormalization
import cv2
import numpy as np
import datetime
import uuid
import os
import shutil
import google.protobuf

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # function -> runs before every request
    allow_origins=["*"], # what domain should we able to talk to our api
    allow_credentials=True, # 
    allow_methods=["*"], # not only allow spicific domain but also we allow specific http method -> if public api -> user get data -> we not allow user to make put request 
    allow_headers=["*"], # spcific header
)

# mount static html file for front end
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define a directory where you want to store the uploaded images
generated_image_dir = "generated_image"

# Check if the directory exists, and if so, delete it
if os.path.exists(generated_image_dir):
    shutil.rmtree(generated_image_dir)

# Create a new directory
os.makedirs(generated_image_dir)

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_monet_to_photo = load_model('Models/g_model_AtoB_001072.h5', cust)
model_photo_to_monet = load_model('Models/g_model_BtoA_001072.h5', cust)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def preprocessing(image):
    # Add preprocessing steps here (e.g., resizing, normalizing, etc.)
    image = image.resize((256, 256))
    image = image.convert("RGB")
    image = img_to_array(image)
    image = (image - 127.5) / 127.5
    return image

def save_image(image_name, image):
     # Save the uploaded image
    image.save(f"{generated_image_dir}{image_name}")


def display_image(image_name, image):
    cv2.imshow("Preprocessed Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

@app.post("/generate_images")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    # Preprocess the image
    preprocessed_image = preprocessing(image)   
    preprocessed_image = np.expand_dims(preprocessed_image, 0)
    
    # Generate monet painting from the photo
    monet_generated = model_photo_to_monet.predict(preprocessed_image)

    # Generate monet painting from the photo
    photo_generated = model_monet_to_photo.predict(monet_generated)

    # Generate unique timestamps and UUIDs
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())

    # Save and display the images with unique filenames
    original_image_filename = f"original_image_{timestamp}_{unique_id}.jpg"
    monet_generated_filename = f"monet_generated_{timestamp}_{unique_id}.jpg"
    photo_generated_filename = f"photo_generated_{timestamp}_{unique_id}.jpg"

    # Save and display the images
    save_image(original_image_filename,image)
    save_image(monet_generated_filename, Image.fromarray(np.uint8(monet_generated[0] * 127.5 + 127.5), 'RGB'))
    save_image(photo_generated_filename, Image.fromarray(np.uint8(photo_generated[0] * 127.5 + 127.5), 'RGB'))

    # Encode string as base64 strings
    with open(f"{generated_image_dir}{monet_generated_filename}","rb") as img_file:
        monet_generated_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    with open(f"{generated_image_dir}{photo_generated_filename}","rb") as img_file:
        photo_generated_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # remove saved file
    os.remove(f"{generated_image_dir}{original_image_filename}")
    os.remove(f"{generated_image_dir}{monet_generated_filename}")
    os.remove(f"{generated_image_dir}{photo_generated_filename}")

    return { "image_1" : monet_generated_base64 , "image_2" : photo_generated_base64 }