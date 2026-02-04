# ArtRestorationAI
Deep learning–based art restoration system using Spatial Transformer and U-Net. Trained on paired distorted and clear images with pixel-wise and perceptual (VGG16) loss. Includes a Streamlit UI to upload damaged images and generate restored outputs in real time.

This project focuses on restoring distorted or damaged images using a deep learning–based image restoration system. The model is trained to convert low-quality, noisy, or damaged artwork images into visually clear and high-quality restored versions.

The system uses a Spatial Transformer Network (STN) combined with a U-Net architecture to automatically align, correct color distortions, and reconstruct missing or degraded visual details. A perceptual loss function based on VGG16 is used along with pixel-wise loss to ensure that restored images are not only accurate at pixel level but also visually realistic.

The model is trained on paired datasets of distorted images and their corresponding clear images. During training, mixed-precision computation and checkpointing are used to optimize memory usage and improve performance. Early stopping and learning rate scheduling are applied to prevent overfitting and improve convergence.

A Streamlit-based user interface is developed to make the model easy to use. Users can upload a distorted image through the UI, and the trained model restores it in real time and displays the output image.

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

#PRE TRAINED MODEL LINK: https://drive.google.com/file/d/1_FJnB5zH128J6QmRX-iMf5pYLHiKkJhl/view?usp=sharing
