import numpy as np

class MaxPooling:
    def __init__(self, pool_size=2) -> None:
       self.pool_size = pool_size


    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // self.pool_size
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j


    def forward(self, input):
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output
    
    def backprop(self, grad_L_out):
        grad_L_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            grad_L_input[i * 2 + i2, j * 2 + j2, f2] = grad_L_out[i, j, f2]

        return grad_L_input