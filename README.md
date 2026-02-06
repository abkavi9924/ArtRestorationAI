# ArtRestorationAI

## One Line Value Proposition
Restores damaged artwork and photos into high quality images using Python + ML; deployable via Streamlit.

## What problem this solves
Manual art restoration is slow and expensive, this project automates the initial steps and work as an assitant to get a view of full restored image.

Demo:
<img width="1867" height="838" alt="image" src="https://github.com/user-attachments/assets/5e6e550b-ffc7-4c70-8991-6b78b24dc7c3" />

<img width="857" height="753" alt="image" src="https://github.com/user-attachments/assets/e4ab414d-54b4-4f05-b4b9-a39725474318" />

<img width="891" height="776" alt="image" src="https://github.com/user-attachments/assets/994a3d8d-ee66-4c11-a8f4-43b4bd885c0a" />

<img width="861" height="746" alt="image" src="https://github.com/user-attachments/assets/93e98663-851b-4dd9-9438-aa9c770253b4" />


**This system can be applied to**:
* Digital art restoration
* Old photograph enhancement
* Museum and archival image preservation
* Damage and noise removal in scanned artworks

**Key Features**
* Spatial Transformer for geometric correction
* U-Net for image restoration
* Perceptual loss using VGG16 for visual quality
* Mixed precision training for faster performance
* Checkpoint-based training recovery
* Streamlit web interface for easy usage

**Technologies Used**
* Python
* PyTorch
* OpenCV
* Torchvision
* Streamlit
* CUDA (for GPU acceleration)
* Dataset used: https://www.kaggle.com/datasets/sankarmechengg/art-images-clear-and-distorted

#PRE TRAINED MODEL LINK: https://drive.google.com/file/d/1_FJnB5zH128J6QmRX-iMf5pYLHiKkJhl/view?usp=sharing

How to run(3 minutes max):

Step 1: Download the pretrained weight from the link above.

Step 2: Download the app.py & models.py file.

Step 3: Install deps

Step 4: Change the location gpu_model with the location of the file saved.

Step 5: Run app.py using **streamlit run app.py**

**Note: Also Download models.py as helper functions from this files are being called in app.py**
