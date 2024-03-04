import numpy as np
from structure.ConvulationLayer import ConvulationLayer
from structure.MaxPoolingLayer import MaxPooling
from structure.Softmax import Softmax

class ConvulationNeuralNetwork():
    def __init__(self) -> None:
        self.conv_layer = ConvulationLayer(8)            # [28, 28, 1] -> [26, 26, 8]
        self.pooling_layer = MaxPooling()                # [26, 26, 8] -> [13, 13, 8]
        self.softmax_layer = Softmax(13 * 13 * 8, 10)    # [13, 13, 8] -> 10
    
    def forward(self, image, label):
        '''
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv_layer.forward((image / 255) - 0.5)
        out = self.pooling_layer.forward(out)
        out = self.softmax_layer.forward(out)

        # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc
    
    def backprop(self, image, label, lr=.005):
        '''
        Completes a full training step on the given image and label.
        Returns the cross-entropy loss and accuracy.
        - image is a 2d numpy array
        - label is a digit
        - lr is the learning rate
        '''
        # Forward
        out, loss, acc = self.forward(image, label)

        # Calculate initial gradient
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # Backprop
        gradient = self.softmax_layer.backprop(gradient, lr)
        gradient = self.pooling_layer.backprop(gradient)
        gradient = self.conv_layer.backprop(gradient, lr)

        return loss, acc
    
    def predict(self, image):
        '''
        Completes a forward pass of the CNN and calculates the accuracy and
        cross-entropy loss.
        - image is a 2d numpy array
        - label is a digit
        '''
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        out = self.conv_layer.forward((image / 255) - 0.5)
        out = self.pooling_layer.forward(out)
        out = self.softmax_layer.forward(out)

        return out.tolist().index(max(out))
    
    def train(self, train_images, train_labels, test_images, test_labels):
        for epoch in range(3):
            print('--- Epoch %d ---' % (epoch + 1))

            # Shuffle the training data
            permutation = np.random.permutation(len(train_images))
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]

            # Train!
            loss = 0
            num_correct = 0
            for i, (im, label) in enumerate(zip(train_images, train_labels)):
                if i % 100 == 99:
                    print(
                        '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, loss / 100, num_correct)
                    )
                    loss = 0
                    num_correct = 0

                l, acc = self.backprop(im, label)
                loss += l
                num_correct += acc

            # Test the CNN
            print('\n--- Testing the CNN ---')
            loss = 0
            num_correct = 0
            for im, label in zip(test_images, test_labels):
                _, l, acc = self.forward(im, label)
                loss += l
                num_correct += acc

            num_tests = len(test_images)
            print(f'Test Loss: {(loss / num_tests)}')
            print(f'Test Accuracy: {(num_correct / num_tests)*100}%')