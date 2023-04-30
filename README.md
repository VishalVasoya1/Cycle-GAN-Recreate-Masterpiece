# CycleGAN for Photo to Monet Painting Generator

🎨 This project contains a CycleGAN implementation for generating monet-style paintings from photos. 

## Steps to Train the Model

### Step 1: CycleGAN Training
* The `CycleGAN_training` folder contains the Python code required to train the CycleGAN model on your own dataset.
* Before running the training code, you can customize the training parameters such as the number of epochs, learning rate, and batch size according to your requirements.
* After the training is complete, the model will be saved to the `Model` folder for later use.

### Step 2: Trained Model
* The `Model` folder contains the trained CycleGAN model for generating monet-style paintings from photos and vice versa.
* You can use this pre-trained model to generate monet-style paintings from your own photos or use it as a starting point for further fine-tuning and experimentation.

## Step 3: Performance

The `performance` folder contains visualizations of the trained model's performance. You can see how well the model is able to convert photos to monet-style paintings and vice versa.

## Step 4: Deployment

This folder `Deployment-FastAPI` contains an API endpoint for the model that I used to deploy my model to Docker and GCP. It also contains a template inside it that contains a webpage for the model.

This folder `utils` contains API endpoints using Flask. Both FastAPI and Flask are available.

**Step 4.1: Deploy to Docker container**

File name: `Dockerfile` and `docker-compose.yml` file contain instructions to deploy the code to a Docker container. Once the Docker container deployment is done locally, I tried to deploy that container on GCP Artifact Registry.

Instructions:

👉 1. Create a repository in GCP in the section of Artifact Registry. Configure the Docker region by using this command in the Google Cloud SDK shell:
```
gcloud auth configure-docker asia-south1-docker.pkg.dev
```
👉 2. Push the Docker container to the repository in GCP Artifact Registry using this command in the Google Cloud SDK shell:
```
docker tag cyclegan-masterpiece:latest asia-south1-docker.pkg.dev/cyclegan-recreate-master-piece/cyclegan-masterpiece/cyclegan-masterpiece:latest 
docker push asia-south1-docker.pkg.dev/cyclegan-recreate-master-piece/cyclegan-masterpiece/cyclegan-masterpiece:latest
```
👉 3. Deploy the container on GCP by running this command in the Google Cloud SDK:
```
gcloud run deploy cyclegan-master \
			--image asia-south1-docker.pkg.dev/cyclegan-recreate-master-piece/cyclegan-masterpiece/cyclegan-masterpiece:latest  \
			--region asia-south1 \
			--port 8000 \
			--memory 8Gi 
```
👉 4. Run the app on the provided URL.
Deployed Model [GitHub Pages](https://cyclegan-recreate-master-piece.el.r.appspot.com/).


**Step 4.2: Deploy to GCP**

Folder named `DeployGCP` contains all the requirements for deploying the model to GCP using App Engine.

👉 Step 1: Open the Google Cloud Shell SDK and run this command to deploy the app using the `app.yaml` file:
```
gcloud app deploy app.yaml --project cyclegan-recreate-master-piece
```
👉 Step 2: Once the deployment is successful, you will get a URL from GCP. Run this URL to access the app.

Note: Make sure to configure the `app.yaml` file properly before deploying the app to GCP. Also, the project ID `cyclegan-recreate-master-piece` is just an example, you need to replace it with your own project ID.

This step-by-step guide will help you deploy your CycleGAN model to GCP with ease.

🚀 Now you are ready to generate monet-style paintings from your own photos using CycleGAN!
