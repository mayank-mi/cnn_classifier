# import numpy as np
# import matplotlib.pyplot as plt
# # Assuming 'C:/demo/classfier_cnn/data\The Eiffel Tower.npy' is the correct path to your .npy file
# array_data = np.load('C:/demo/classfier_cnn/data/The Eiffel Tower.npy')

# # Get the shape of the array
# shape_of_array = np.shape(array_data)

# # print(shape_of_array)



# # Load the data
# array_data = np.load('C:/demo/classfier_cnn/data/The Eiffel Tower.npy')

# # Extract the first image
# first_image = array_data[0].reshape(28, 28)  # Assuming each image is 28x28 pixels (adjust as per your data)

# # Display the first image
# plt.imshow(first_image, cmap='gray')
# plt.axis('off')  # Optional: turn off axis numbers and ticks
# plt.show()

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,'89989')