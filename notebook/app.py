import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import io

# Initialize Flask app
app = Flask(__name__)

# Define class labels and class dictionary
class_labels = [
    'The Eiffel Tower',
    'full_numpy_bitmap_basketball',
    'full_numpy_bitmap_baseball',
    'full_numpy_bitmap_bathtub',
    'full_numpy_bitmap_bicycle',
    'full_numpy_bitmap_apple',
    'full_numpy_bitmap_bat',
    'full_numpy_bitmap_alarm clock',
    'full_numpy_bitmap_airplane',
    'full_numpy_bitmap_book'
]

class_dict = {i: label for i, label in enumerate(class_labels)}

# Define the model
class DoodleClassModel04(nn.Module):
    def __init__(self, input_channels: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=output_shape)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.layer_stack(x)

# Initialize and load the model
input_channels = 1
output_shape = 10
model = DoodleClassModel04(input_channels=input_channels, output_shape=output_shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load('C:/test001/ttt/cnn_classifier/ml_model/cnn_model_2307.pth', map_location=device))
model.eval()

# Preprocessing function
def preprocess_sample_image(image_array):
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image_array = np.mean(image_array, axis=2)
    if image_array.shape != (28, 28):
        raise ValueError(f"Expected image shape (28, 28), got {image_array.shape}")
    image_array = image_array.astype('float32') / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.to(device)
    return image_tensor

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    print(file,'5555555')
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('L')
        resized_image = image.resize((28, 28))
        image_array = np.array(resized_image)
        sample_image_tensor = preprocess_sample_image(image_array)

        with torch.no_grad():
            output = model(sample_image_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            return jsonify({
                'predicted_class_label': class_dict[predicted_class],
                'predicted_class_index': predicted_class
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)


# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# import numpy as np
# from PIL import Image

# class_labels = [
#     'The Eiffel Tower',
#     'full_numpy_bitmap_basketball',
#     'full_numpy_bitmap_baseball',
#     'full_numpy_bitmap_bathtub',
#     'full_numpy_bitmap_bicycle',
#     'full_numpy_bitmap_apple',
#     'full_numpy_bitmap_bat',
#     'full_numpy_bitmap_alarm clock',
#     'full_numpy_bitmap_airplane',
#     'full_numpy_bitmap_book'
# ]

# # Create the dictionary
# class_dict = {i: label for i, label in enumerate(class_labels)}

# class DoodleClassModel04(nn.Module):
#     def __init__(self, input_channels: int, output_shape: int):
#         super().__init__()
#         self.layer_stack = nn.Sequential(
#             nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.4),

#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.4),

#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.4),

#             nn.Flatten(),
#             nn.Linear(in_features=128*3*3, out_features=256),
#             nn.ReLU(),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.4),
#             nn.Linear(in_features=256, out_features=output_shape)
#         )
    
#     def forward(self, x):
#         # Add channel dimension if it's missing
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         return self.layer_stack(x)
    
# # Define the input channels and output shape
# input_channels = 1  # Assuming grayscale images
# output_shape = 10  # Assuming 10 classes for classification

# # Initialize the model
# model = DoodleClassModel04(input_channels=1, output_shape=len(class_dict))

# # Check if CUDA is available and move the model to GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model03 = model.to(device)

# # Load the model weights
# model03.load_state_dict(torch.load('C:/test001/ttt/cnn_classifier/ml_model/cnn_model_2307.pth', map_location=device))

# # Set the model to evaluation mode
# model03.eval()
# ################################################

# def preprocess_sample_image(image_array):
#     # Check the shape of the input array
#     if image_array.ndim == 3 and image_array.shape[2] == 3:
#         # If it's a color image, convert to grayscale
#         image_array = np.mean(image_array, axis=2)
    
#     # Ensure the image is 28x28
#     if image_array.shape != (28, 28):
#         raise ValueError(f"Expected image shape (28, 28), got {image_array.shape}")
    
#     # Normalize the image
#     image_array = image_array.astype('float32') / 255.0
    
#     # Convert to torch tensor
#     image_tensor = torch.tensor(image_array, dtype=torch.float32)
    
#     # Add batch and channel dimensions
#     image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, 28, 28)
    
#     # Check if CUDA (GPU) is available and move tensor to CUDA device
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#         image_tensor = image_tensor.to(device)
    
#     return image_tensor

# # Load and preprocess the image
# image = Image.open('C:/test001/ttt/cnn_classifier/iamges/apple.jpg')

# # Resize the image to 28x28
# resized_image = image.resize((28, 28))

# # Convert the image to a NumPy array
# image_array = np.array(resized_image)

# sample_image_tensor = preprocess_sample_image(image_array)

# # array_data = np.load('C:/demo/classfier_cnn/data/full_numpy_bitmap_bicycle.npy')

# # # Extract the first image
# # image_array = array_data[16].reshape(28, 28)
# # sample_image_tensor = preprocess_sample_image(image_array)

# # Make a prediction with the loaded model
# with torch.no_grad():
#     output = model03(sample_image_tensor)
#     predicted_class = torch.argmax(output, dim=1).item()

# # Print the predicted class
# print(f"Predicted class label: {class_dict[predicted_class]}")
# print(f"Predicted class index: {predicted_class}")