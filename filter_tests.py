import numpy as np
from PIL import Image

class ConvulationLayer():
    def __init__(self, filter) -> None:
        filter_height, filter_width = filter.shape

        self.filter = filter
        self.pad_height, self.pad_width = filter_height // 2, filter_width // 2
       

    def forward(self, input_image):
        image_height, image_width, image_channels = input_image.shape
        output = np.zeros((image_height - 2*self.pad_height, image_width - 2*self.pad_width, image_channels))

        for c in range(image_channels):
            for y in range(self.pad_height, image_height - self.pad_height):
                for x in range(self.pad_width, image_width - self.pad_width):
                    output[y-self.pad_height, x-self.pad_width, c] = np.sum(input_image
                                [y-self.pad_height:y+self.pad_height+1, 
                                x-self.pad_width:x+self.pad_width+1, c] * self.filter)
        return output

class MaxPoolingLayer():
    def __init__(self, pool_size) -> None:
        self.pool_size = pool_size

    def forward(self, input_image):
        image_height, image_width, image_channels = input_image.shape
        new_height, new_width = image_height // self.pool_size, image_width // self.pool_size
        output = np.zeros((new_height, new_width, image_channels))

        for c in range(image_channels):
            for y in range(0, new_height):
                for x in range(0, new_width):
                    output[y, x, c] = np.max(input_image
                            [y*self.pool_size:(y+1)*self.pool_size,
                              x*self.pool_size:(x+1)*self.pool_size, c])
        return output


class ConvulationNetwork():
    def __init__(self, filter, pool_size) -> None:
        self.layers = [
            ConvulationLayer(filter),
        ]
    
    def forward(self, input_image):
        output = input_image

        for layer in self.layers:
            print(output.shape)
            output = layer.forward(output)
        
        return output

filter = np.array([[0, 0, 0],
                    [1, 5, 1],
                    [0, 0, 0]])

image = np.array(Image.open('tests/image.jpg'))
model = ConvulationNetwork(filter, 2)

output = model.forward(image)
Image.fromarray(output.astype(np.uint8)).save('tests/new_image.jpg')

# Zapis obrazów wynikowych
# Image.fromarray(convolved_image.astype(np.uint8)).save('convolved_image.jpg')
# Image.fromarray(pooled_image.astype(np.uint8)).save('pooled_image.jpg')
