import numpy as np
from structure.ConvulationLayer import ConvulationLayer
from structure.MaxPoolingLayer import MaxPooling
from structure.Softmax import Softmax

class ConvulationNeuralNetwork():
    def __init__(self, load_model, save_model) -> None:
        # Dev Tools        
        self.load_model = load_model
        self.save_model = save_model

        # Hyperparameters
        self.conv_layer = ConvulationLayer(8)
        self.pooling_layer = MaxPooling()                
        self.softmax_layer = Softmax(13 * 13 * 8, 10)    

        if self.load_model:
            self.softmax_layer.load_parameters()
    
    def forward(self, image, label):
        out = self.conv_layer.forward((image / 255) - 0.5)
        out = self.pooling_layer.forward(out)
        out = self.softmax_layer.forward(out)

        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc
    
    def backprop(self, image, label, lr=.005):
        out, loss, acc = self.forward(image, label)

        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        gradient = self.softmax_layer.backprop(gradient, lr)
        gradient = self.pooling_layer.backprop(gradient)
        gradient = self.conv_layer.backprop(gradient, lr)

        return loss, acc
    
    def predict(self, image):
        out = self.conv_layer.forward((image / 255) - 0.5)
        out = self.pooling_layer.forward(out)
        out = self.softmax_layer.forward(out)

        return out.tolist().index(max(out))
    
    def train(self, train_images, train_labels, test_images, test_labels):
        for epoch in range(3):
            print('--- Epoch %d ---' % (epoch + 1))

            permutation = np.random.permutation(len(train_images))
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]

            loss = 0
            num_correct = 0
            for i, (im, label) in enumerate(zip(train_images, train_labels)):
                if i % 100 == 99:
                    print(
                        '[Epoch %d]: Average Loss %.3f | Accuracy: %d%%' %
                        (i + 1, loss / 100, num_correct)
                    )
                    loss = 0
                    num_correct = 0
                    if self.save_model:
                        self.softmax_layer.save_parameters()

                l, acc = self.backprop(im, label)
                loss += l
                num_correct += acc

            loss = 0
            num_correct = 0
            for im, label in zip(test_images, test_labels):
                _, l, acc = self.forward(im, label)
                loss += l
                num_correct += acc

            num_tests = len(test_images)
            print(f'Test Loss: {(loss / num_tests)}')
            print(f'Test Accuracy: {(num_correct / num_tests)*100}%')
    
    def test(self, test_images, test_labels):
        loss = 0
        num_correct = 0
        for im, label in zip(test_images, test_labels):
            _, l, acc = self.forward(im, label)
            loss += l
            num_correct += acc
        
        num_tests = len(test_images)
        print(f'Test Accuracy: {(num_correct / num_tests)*100}%')