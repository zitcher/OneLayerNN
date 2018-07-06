import numpy as np


def l2_loss(Y, predictions):
    '''
        Sum of the squared losss inputs Y and predictions.
    '''
    return np.sum(np.power(Y-predictions, 2))


class OneLayerNN:
    '''
        One layer NN trained with gradient descent
    '''
    def __init__(self):
        '''
            sets weights of the NN
        '''
        self.weights = None
        pass

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using gradient descent
        '''
        # TODO
        self.weights = np.zeros(X.shape[1])

        for i in range(epochs):
            # random order for better predictions
            in_out = self.shuffle_two(X, Y)
            data = in_out[0]
            label = in_out[1]
            for j in range(len(label)):
                data_j = data[j]
                label_j = label[j]
                gradient = -2 * (label_j - self.sum_weight_input(data_j)) * data_j
                self.weights -= learning_rate*gradient

    # shuffles both arrays in tandem
    def shuffle_two(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def sum_weight_input(self, X):
        return np.dot(X, self.weights)

    def predict(self, X):
        '''
        Returns predictions of the NN on a set of examples X.
        '''
        # TODO
        predictions = np.zeros(len(X))
        for i in range(0, len(X)):
            predictions[i] = self.predictInput(X[i])

        return predictions

    def predictInput(self, a_input):
        return self.sum_weight_input(a_input)

    def loss(self, X, Y):
        '''
        Returns the squared error
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X 2D Numpy array where each row contains an example
        :param Y 1D Numpy array containing the corresponding values for each example
        :return A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
