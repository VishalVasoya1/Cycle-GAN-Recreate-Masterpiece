from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import tensorflow as tf
from keras.models import load_model
from flask_cors import CORS
from tensorflow_addons.layers import InstanceNormalization
from keras.utils import img_to_array
from keras.utils import load_img
app = Flask(__name__)
CORS(app)

# Load the TensorFlow model
model_1 = load_model('Models/g_model_AtoB_001000.h5')
model_2 = load_model('Models/g_model_BtoA_001000.h5')

# Define the routes for the app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Get the uploaded file from the request
    file = request.files['file']

    # Open the uploaded file with PIL and convert to RGB
    img = Image.open(file)
    img = img.convert('RGB')
    # Return the image data as a byte string
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    return img_bytes

def preprocessing(img):
    input_image = img_to_array(img)
    input_image = (input_image - 127.5) / 127.5
    input_image = tf.expand_dims(input_image, 0)
    return input_image

@app.route('/generate', methods=['POST'])
def generate():
    # Get the uploaded file from the request
    file = request.files['file']

    # Open the uploaded file with PIL and convert to RGB
    img = Image.open(file)
    img = img.convert('RGB')
    img = img.resize((256,256))
    input_image = preprocessing(img)
    print(input_image.shape)
    # Convert the PIL image to a TensorFlow tensor

    generated_image_1 = model_1(input_image)[0]
    generated_image_1 = tf.image.convert_image_dtype(generated_image_1, tf.uint8)
    print("Generated 1: ",generated_image_1.shape)
    generated_image_1 = Image.fromarray(generated_image_1.numpy())
    
    
    generated_image_2 = model_2(input_image)[0]
    generated_image_2 = tf.image.convert_image_dtype(generated_image_2, tf.uint8)
    print("Generated 2: ",generated_image_2.shape)
    generated_image_2 = Image.fromarray(generated_image_2.numpy())
    
    img1_bytes = BytesIO()
    img2_bytes = BytesIO()
    generated_image_1.save(img1_bytes, format='JPEG')
    generated_image_2.save(img2_bytes, format='JPEG')
    img1_bytes.seek(0)
    img2_bytes.seek(0)

    # # Return the two image data as byte strings
    return {"image_1": img1_bytes.read(), "image_2": img2_bytes.read()}

if __name__ == '__main__':
    app.run()