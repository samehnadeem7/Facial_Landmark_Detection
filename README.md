# Facial Landmark Detection

## OBJECTIVE
The goal of the project is to develop a highly efficient model capable of accurately detecting facial key points in input images. This model will support real-time inference for applications such as face recognition, emotion detection, virtual makeup, and augmented reality (AR).

## LEARNING PROCESS

![Learning Process](https://github.com/user-attachments/assets/d39e5d67-d01e-4e4d-85c0-ccb578650642)

1. Started with Andrew Ng's Deep Learning course to grasp the fundamentals.
2. Worked on MNIST using Numpy, implementing basic ML without libraries.
3. Applied CNN to MNIST, moving towards deeper architectures.
4. Switched to PyTorch for MNIST, gaining hands-on experience with this framework.
5. Tackled CIFAR-10 in PyTorch, handling a more complex dataset.
6. Created a dataset class for BSD-100, enhancing my data pipeline skills.
7. Moved to facial landmark detection, working on identifying key points in facial images.

### MNIST-HANDWRITTEN DIGITS

The MNIST AND MNIST-HANDWRITTEN DIGITS datasets are used. These datasets contain 60,000 training samples and 10,000 test samples. Each sample in the MNIST dataset is a 28x28 pixel grayscale image of a single handwritten digit between 0 & 9.

![MNIST Digits](https://hackmd.io/_uploads/ByroZQK11x.png)

| **Parameter**                   | **Description**                        |
|----------------------------------|----------------------------------------|
| Used neural network              | Single layer neural network            |
| Number of iterations             | 2500                                   |
| Gradient descent method          | Full batch gradient descent            |
| Iterations = Epochs              | Yes                                    |
| Activation functions tried       | ReLU/Tanh with softmax combination     |
| Loss plot                        | Loss (cost) vs iteration (epochs)      |


#### RESULTS & Accuracy:
![MNIST Digits](https://github.com/user-attachments/assets/37e8bc80-e9b3-40cf-9169-c3258374ad50)

![Results](https://github.com/user-attachments/assets/ae1ae492-85e0-4aff-992f-26d7a4cc3986)



Visualization:

![Visualization](https://hackmd.io/_uploads/SJPVGXFkJx.png)

Graphs:

![Graphs](https://github.com/user-attachments/assets/19a40f0d-406d-4b32-bc3c-76cb31a64d60)


### CIFAR-10

The CIFAR-10 dataset is used. This dataset contains 60,000 samples, with 50,000 training samples and 10,000 test samples. Each sample in the CIFAR-10 dataset is a 32x32 pixel color image belonging to one of 10 different classes, such as airplanes, cars, birds, cats, and more.

![CIFAR-10](https://hackmd.io/_uploads/H1Ub4XKkkx.png)

#### HYPERPARAMETERS

| Hyperparameters | Value | 
| -------- | -------- | 
| Batch-Size | 1024 | 
| Learning-rate | 0.0001 |
| Num of Epochs | 100 |
| Loss | Cross Entropy Loss |
| Optimizer | Adam Optimizer |

#### RESULTS 


![Accuracy](https://github.com/user-attachments/assets/06c50e68-0770-4c55-9ecb-a8be9dfec28d)


![Graphs](https://github.com/user-attachments/assets/ab3e58b9-1176-4335-911d-fdc5c8bb1a4e)

### CUSTOM DATASET

The dataset is sourced from the iBUG 300-W dataset with XML-based annotations for facial landmarks and crop coordinates. It extracts image paths, 68 landmark points, and face cropping coordinates. The preprocessing pipeline resizes images with padding, applies random augmentations such as color jitter, offset and random cropping, and random rotation, adjusting landmarks accordingly. Landmarks are normalized relative to image dimensions, and images are converted to grayscale before being transformed into tensors normalized between [-1, 1]. A dataset class handles parsing, preprocessing, and returning ready-to-use images and normalized landmarks.

#### RESULTS

Output:
![Output](https://github.com/user-attachments/assets/5a7a30b2-4baf-4701-81e2-a2a59e63bcc7)
#### visualisation

## MAIN PROJECT
### FACIAL LANDMARK DETECTION

For facial landmark detection, datasets such as the *iBUG 300-W* and similar facial landmark datasets are commonly used. These datasets contain a variety of images annotated with facial key points. For example, the *iBUG 300-W* dataset consists of thousands of images, with each image labeled with 68 facial landmarks, including points around the eyes, nose, mouth, and jawline. The dataset is typically divided into training and test sets, enabling model training and evaluation for tasks like face recognition, emotion analysis, and augmented reality applications.

#### PROCEDURE

In this project, a **FaceLandmarksAugmentation** class was implemented to apply various image augmentation techniques, such as cropping, random cropping, random rotation, and color jittering, specifically tailored for facial landmark detection.

![Augmentation](https://hackmd.io/_uploads/HkrIMVtk1l.png)

This class initializes key parameters like image dimensions, brightness, and rotation limits. Methods such as `offset_crop` and `random_rotation` adjust landmark coordinates accordingly. A **Preprocessor** class initializes the augmentation methods and normalizes the data, while a **datasetlandmark** class, inheriting from `Dataset`, handles image paths, landmark coordinates, and cropping information parsed from XML files. The dataset is split into training and validation sets, with `DataLoader` objects created for each, and a function is defined to visualize images with corresponding landmarks.

![Visualization](https://hackmd.io/_uploads/HJco7EYy1e.png)

The **network design** is a modular convolutional neural network (CNN) with depthwise separable convolutions to improve efficiency. It includes an entry block for initial feature extraction, middle blocks with residual connections, and an exit block that outputs facial landmark coordinates. The CNN uses batch normalization and LeakyReLU for better performance. A training loop is implemented to run for 30 epochs, where the model computes loss, updates weights using an optimizer, validates after each epoch, and saves checkpoints to prevent overfitting and retain the best-performing model.

![CNN Network](https://hackmd.io/_uploads/BJo3HEYJkl.png)

#### HYPERPARAMETERS

| Hyperparameters | Value | 
| -------- | -------- | 
| Batch-Size | 32 |
| Learning-rate | 0.00075 |
| Num of Epochs | 30 |
| Loss | MSE LOSS |
| Optimizer | Adam Optimizer |

#### RESULTS

LOSS:

Loss at epoch 1:
![Loss1](https://hackmd.io/_uploads/S1kkuNKJ1l.png)

Loss at epoch 2:
![Loss2](https://hackmd.io/_uploads/H1C-OEKykg.png)

Loss at epoch 3:
![Loss3](https://hackmd.io/_uploads/HJfr_4Kyyx.png)

GRAPHS:
![Graphs](https://hackmd.io/_uploads/S1qu_VYkyg.png)

### SOFTWARE TOOLS USED

- Python
- Numpy
- PyTorch
- PIL
- Matplotlib
