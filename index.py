from keras.datasets import mnist
from structure.ConvulationNetwork import ConvulationNeuralNetwork
from PIL import Image
import numpy as np

(train_X, train_y), (test_X, test_y) = mnist.load_data()

load_model = False
save_model = True

model = ConvulationNeuralNetwork(load_model, save_model)

model.train(train_X[:1000], train_y[:1000], test_X[:1000], test_y[:1000])
model.test(test_X[:1000], test_y[:1000])

BASE_TEST_PATH = "data/manual_test"
TEST_PATHS = [
    f"{BASE_TEST_PATH}/img_26269.jpg",
    f"{BASE_TEST_PATH}/img_26270.jpg",
    f"{BASE_TEST_PATH}/img_26271.jpg",
    f"{BASE_TEST_PATH}/img_26272.jpg",
    f"{BASE_TEST_PATH}/img_26273.jpg",
    f"{BASE_TEST_PATH}/img_26274.jpg",
    f"{BASE_TEST_PATH}/paint_test.jpg"
]

manual_test_data = []

for path in TEST_PATHS:
    image = np.array(Image.open(path))

    if len(image.shape) == 3:
        gray_array = image[:, :, 0]
        image = gray_array
    manual_test_data.append([path, image])

for (path, image_test) in manual_test_data:
    output = model.predict(image_test)
    print(f"Image Test: {path}: {output}")

while True:
    path = input("Enter path for the image: ")

    image = np.array(Image.open(path))
    if len(image.shape) == 3:
        gray_array = image[:, :, 0]
        image = gray_array
    
    output = model.predict(image)
    print(f'This is {output}')