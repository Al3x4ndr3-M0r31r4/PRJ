import numpy as np


class SyntheticDataset:

    def __init__(self, min_val, max_val, num_rows, num_features, num_classes, random_coef=None, norm_dist=False, class_percent=None, seed=None,
                 std_dev_perc=None, class_sep_perc=None):
        """
        :param max_val: maximum value of the random generated values
        :param min_val: minimum value of the random generated values
        :param num_rows: Number of rows
        :param num_features: Number of features
        :param num_classes: Number of classes
        :param random_coef: Array with the randomness coefficient of each column with domain [0, 1]
        :param class_percent: Array with probabilities of the classes in the domain ]0, 1]. If None, all classes will have same probabilities
        :param seed: seed for the random generator
        :param std_dev_perc: the standard deviation for each feature when using norm_dist=True. All gaussian curves of the same feature have the
        same standard deviation
        :param class_sep_perc: The percentage of separation for each feature. The greater the separation the easier should be the classification and
        the curves of the feature will be more apart if the hystogram is analysed
        """

        self.min_val = min_val
        self.max_val = max_val
        self.num_rows = num_rows
        self.num_features = num_features
        self.num_classes = num_classes
        self.random_coef = random_coef
        self.norm_dist = norm_dist
        self.std_dev_perc = std_dev_perc
        self.class_sep_perc = class_sep_perc
        self._dataset = None
        self._x = None
        self._y = None

        if class_percent is None:
            self.class_percent = [1 / num_classes for _ in range(num_classes)]
        else:
            self.class_percent = class_percent

        if seed is not None:
            self.rand_gen = np.random.default_rng(seed=seed)
        else:
            self.rand_gen = np.random.default_rng()

        if norm_dist and std_dev_perc is None:
            self.std_dev_perc = np.ones(num_features, dtype=float)

        if norm_dist and class_sep_perc is None:
            self.class_sep_perc = np.ones(num_features, dtype=float)

        self.check_errors()


    def generate(self):
        if not self.norm_dist:
            self.generate_unif_dist()
        else:
            self.generate_norm_dist()

        self._x = np.array(self._dataset[:, :-1])
        self._y = np.array([int(x) for x in self._dataset[:, -1]])

    def generate_unif_dist(self):
        self._dataset = self.rand_gen.uniform(self.min_val, high=self.max_val,
                                              size=(self.num_rows, self.num_features + 1))

        class_samples = [int(np.ceil(x * self.num_rows)) for x in self.class_percent]

        current_idx = 0
        for i in range(len(class_samples)):
            next_idx = min(self.num_rows, current_idx + class_samples[i])
            self._dataset[current_idx: next_idx, -1] = [i for _ in range(next_idx - current_idx)]
            current_idx = next_idx

        rand_prob = self.rand_gen.random(size=(self.num_rows, self.num_features))

        feat_interval = self.max_val - self.min_val
        class_interval = feat_interval / (self.num_classes - 1)

        for i in range(self.num_features):
            for j in range(self.num_rows):
                rand_val = rand_prob[j, i]
                class_val = int(self._dataset[j, -1])
                if rand_val > self.random_coef[i][class_val]:
                    self._dataset[j, i] = self.min_val + (class_val * class_interval)


    def generate_norm_dist(self):
        self._dataset = np.zeros((self.num_rows, self.num_features + 1), dtype=float)

        feat_interval_val = self.max_val - self.min_val
        class_interval_val = feat_interval_val / self.num_classes

        #Calculate the number of samples for each class
        class_samples = [int(np.ceil(x * self.num_rows)) for x in self.class_percent]

        #Calculate the means of the gaussian distribution for each feature for each class
        feat_norm_dist_means = []

        mean = (feat_interval_val / 2) + self.min_val

        for i in range(self.num_features):

            feat_i_means = []

            for x in range(self.num_classes):
                #Mean of the classs "x" for feature "i"
                mean_feat_class =  class_interval_val * (x + 1/2) + self.min_val

                #Add the class separation factor
                feat_i_means.append(
                    mean_feat_class + ((mean - mean_feat_class) * (1 - self.class_sep_perc[i])))

            feat_norm_dist_means.append(feat_i_means)

        #Calculate the standard deviation of each gaussian distribution for each feature
        std_dev_val = self.std_dev_perc * class_interval_val / 3

        for x in range(self.num_features):

            current_idx = 0

            for i in range(len(class_samples)):
                next_idx = min(self.num_rows, current_idx + class_samples[i])
                interval = next_idx - current_idx

                # Create "interval" instances of the class "i"
                self._dataset[current_idx: next_idx, -1] = [i for _ in range(interval)]

                self._dataset[current_idx: next_idx, x] = self.rand_gen.normal(loc=feat_norm_dist_means[x][i], scale=std_dev_val[x], size=interval)

                current_idx = next_idx

            feat_min = np.min(self._dataset[:, x])

            # Put the minimum value to 0 to perform scale operations
            dataset_zero = self._dataset[:, x] - feat_min

            # Get the maximum value
            feat_max_pos_value = np.max(dataset_zero)

            if feat_max_pos_value > feat_interval_val:

                # Scale the feature to its proper interval
                feat_normalize = (self._dataset[:, x] - feat_min) / feat_max_pos_value * feat_interval_val

                # Put the value of the feature back to the original range
                self._dataset[:, x] = feat_normalize + self.min_val


    def check_errors(self):
        try:
            class_sum = np.sum(self.class_percent)
            if round(class_sum, 5) != 1:
                raise ValueError("The sum of class percentages must be equal to 1")
        except:
            raise ValueError("class_percent must be an array of floats within ]0, 1]")

        if len(self.class_percent) != self.num_classes:
            raise ValueError("class_percent must have as many elements as the number of classes")

        if 0 in self.class_percent:
            raise ValueError("All probabilites of the classes must be greater than 0")

        if self.num_rows < self.num_classes:
            raise ValueError("Insufficient rows for the given number of classes")

        if self.norm_dist and len(self.std_dev_perc) != self.num_features:
            raise ValueError("The percentual standard deviation (std_dev_perc) should have as many elements as there are features")

        if self.norm_dist and len(self.class_sep_perc) != self.num_features:
            raise ValueError("The percentual separation of classes (class_sep_perc) should have as many elements as there are features")

        if self.random_coef is not None:
            try:
                rand_coef_vals = np.array(self.random_coef)
                if (rand_coef_vals.shape[1] != self.num_classes):
                    raise ValueError("Invalid random coefficient Shape. The probability of all classes must be provided.")

                elif (rand_coef_vals.shape[0] != self.num_features):
                    raise ValueError("Invalid random coefficient Shape. The probability of all features must be provided.")
            except:
                raise ValueError("Invalid random coefficient")


    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_data(self):
        return self._dataset
