import numpy as np


class SyntheticDataset:

    def __init__(self, min_val, max_val, num_rows, num_features, num_classes, random_coef, seed=None):
        """
        :param max_val: maximum value of the random generated values
        :param min_val: minimum value of the random generated values
        :param num_rows: Number of rows
        :param num_features: Number of features
        :param num_classes: Number of classes
        :param random_coef: Array with the randomness coefficient of each column [0., 1.]
        """

        self.min_val = min_val
        self.max_val = max_val
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_classes = num_classes
        self.random_coef = random_coef
        self._dataset = None

        if seed is not None:
            np.random.seed(seed)

    def generate(self):
        try:
            rand_coef_vals = np.array(self.random_coef)
            if(rand_coef_vals.shape[1] != self.num_classes):
                raise ValueError("Invalid random coefficient Shape. The probability of all classes must be provided.")
        except:
            raise ValueError("Invalid random coefficient")

        self._dataset = np.random.randint(self.min_val, high=self.max_val, size=(self.num_rows, self.num_features + 1))
        self._dataset[:, -1] = [i % self.num_classes for i in range(self.num_rows)]

        for i in range(len(self.random_coef)):
            for j in range(self.num_rows):
                rand_val = np.random.rand()
                class_val = self._dataset[j, -1]
                odd = -1 if i % 2 == 0 else 1
                if rand_val > self.random_coef[i][class_val]:
                    self._dataset[j, i] = (i + 1) * odd * 2 * class_val

    def get_x(self):
        return self._dataset[:, :-1]

    def get_y(self):
        return self._dataset[:, -1]

    def get_data(self):
        return self._dataset
