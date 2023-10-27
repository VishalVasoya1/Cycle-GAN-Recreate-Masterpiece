"""
Cycle GAN: Monet2Photo

Dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

"""

# monet2photo
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)


# dataset path
path = 'monet2photo/'

# load dataset A - Monet paintings
monet_all = load_images(path + 'trainA/')
print('Loaded monet: ', monet_all.shape)

from sklearn.utils import resample
#To get a subset of all images, for faster training during demonstration
monet = resample(monet_all, 
                 replace=False,     
                 n_samples=500,    
                 random_state=42) 

# load dataset B - Photos 
photo_all = load_images(path + 'trainB/')
print('Loaded photo: ', photo_all.shape)
#Get a subset of all images, for faster training during demonstration
#We could have just read the list of files and only load a subset, better memory management. 
photo = resample(photo_all, 
                 replace=False,     
                 n_samples=500,    
                 random_state=42) 

# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(monet[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(photo[i].astype('uint8'))
plt.show()



# load image data
data = [monet, photo]

print('Loaded', data[0].shape, data[1].shape)

#Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
#And tanh ranges between -1 and 1
def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from CycleGAN_model import define_generator, define_discriminator, define_composite_model, train
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_monet_to_photo = define_generator(image_shape)
# generator: B -> A
g_model_photo_to_monet = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_monet = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_photo = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_monet_to_photo = define_composite_model(g_model_monet_to_photo, d_model_photo, g_model_photo_to_monet, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_photo_to_monet = define_composite_model(g_model_photo_to_monet, d_model_monet, g_model_monet_to_photo, image_shape)

from datetime import datetime 
start1 = datetime.now() 
# train models
train(d_model_monet, d_model_photo, g_model_monet_to_photo, g_model_photo_to_monet, c_model_monet_to_photo, c_model_photo_to_monet, dataset, epochs=1000)

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

############################################

# Use the saved cyclegan models for image translation
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = vstack((imagesX, imagesY1, imagesY2))
	titles = ['Real', 'Generated', 'Reconstructed']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()

# load dataset
monet_data = resample(monet_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

B_data = resample(photo_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=42) # reproducible results

monet_data = (monet_data - 127.5) / 127.5
B_data = (B_data - 127.5) / 127.5


# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_monet_to_photo = load_model('monet2photo_models/g_model_monet_to_photo_005935.h5', cust)
model_photo_to_monet = load_model('monet2photo_models/g_model_photo_to_monet_005935.h5', cust)

# plot A->B->A (Monet to photo to Monet)
A_real = select_sample(monet_data, 1)
B_generated  = model_monet_to_photo.predict(A_real)
A_reconstructed = model_photo_to_monet.predict(B_generated)
show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B (Photo to Monet to Photo)
B_real = select_sample(B_data, 1)
A_generated  = model_photo_to_monet.predict(B_real)
B_reconstructed = model_monet_to_photo.predict(A_generated)
show_plot(B_real, A_generated, B_reconstructed)

##########################
#Load a single custom image
test_image = load_img('monet2photo/sunset256.jpg')
test_image = img_to_array(test_image)
test_image_input = np.array([test_image])  # Convert single image to a batch.
test_image_input = (test_image_input - 127.5) / 127.5

# plot B->A->B (Photo to Monet to Photo)
monet_generated  = model_photo_to_monet.predict(test_image_input)
photo_reconstructed = model_monet_to_photo.predict(monet_generated)
show_plot(test_image_input, monet_generated, photo_reconstructed)
