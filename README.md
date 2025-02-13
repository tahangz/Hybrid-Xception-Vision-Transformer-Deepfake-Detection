# Hybrid-Xception-Vision-Transformer-Deepfake-Detection
A deepfake detection system combining Xception and Vision Transformer (ViT) models to identify AI-generated images and videos. Trained on the DFDC dataset, this project leverages TensorFlow/Keras for robust detection of facial manipulation artifacts.

## Dataset 
from kaggle , 140k-real-and-fake-faces
link : https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

## Features  
- Image-based deepfake detection.  
- Video processing by analyzing frames.  
- Hybrid CNN + Transformer architecture for robustness.  
- Easy integration with TensorFlow/Keras.
  
## Results  
- Validation Accuracy: **92.5%**  
- Validation AUC: **0.98**  
- Test Accuracy: **89.7%** 

![Screenshot 2025-02-05 153246](https://github.com/user-attachments/assets/cecdf0f0-c212-4ebe-ac3f-02a877ed9d70)
![Screenshot 2025-02-05 153306](https://github.com/user-attachments/assets/756f067b-0b72-4d3d-a4e5-2799b8e4ec93)
![Screenshot 2025-02-05 153322](https://github.com/user-attachments/assets/510ff550-4e55-41bf-9002-3c3914e29f99)
![Screenshot 2025-02-05 153345](https://github.com/user-attachments/assets/2ed33661-a0a4-4c1d-bbc4-c35325d5a1d4)
![Screenshot 2025-02-05 153410](https://github.com/user-attachments/assets/bdd65dce-edc3-4ad8-8d12-5d977682f120)
![Screenshot 2025-02-05 153443](https://github.com/user-attachments/assets/4ad48bb4-20d1-4466-8b22-6acdad77b842)


## Installation  
1. Clone the repository:  
```bash
git clone https://github.com/tahangz/Hybrid-Xception-Vision-Transformer-Deepfake-Detection.git
cd Hybrid-Xception-Vision-Transformer-Deepfake-Detection
````
## Install dependencies

```bash
pip install tensorflow opencv-python mtcnn pandas tqdm
```
## Download the pre-trained model
Download deepfake_detector.keras and place it in the model folder.

## Usage
# Image Detection
````python
from keras.models import load_model
from utils import predict_image  

model = load_model('model/deepfake_detector.keras')
result = predict_image("test_image.jpg")  # Replace with your image path
print(f"Prediction: {result}")  # Output: "Fake" or "Real"
````

