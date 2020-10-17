import abc
import json

import numpy as np

class Base_Discriminator:
    __metaclass__ = abc.ABCMeta

    def __init__(self, train_data=None,
                 test_data=None,
                 prefix_path=None,
                 bootstrap_sampling=None,
                 retrain=False):
        self.weights_path = prefix_path
        self.history = None
        self.model = self.build_model()

        if bootstrap_sampling:
            print("Bootstrap sampling with data size {}".format(bootstrap_sampling))
            x_train, y_train = train_data
            assert bootstrap_sampling < len(x_train)
            sample_idx = np.arange(len(x_train))
            np.random.shuffle(sample_idx)
            sample_idx = sample_idx[:bootstrap_sampling]
            train_data = x_train[sample_idx], y_train[sample_idx]

        try:
            if retrain:
                raise
            self.history = json.load(open(prefix_path + "_history", 'r'))
            self.model.load_weights(prefix_path)
            self.compile_model()
        except Exception as e:
            print(e)
            if train_data:
                self.train(train_data, test_data)
                if prefix_path:
                    self.model.save_weights(prefix_path)
                    json.dump(str(self.history), open(prefix_path + "_history", 'w'))
                    print("Saved model and history to {}".format(prefix_path))

    def build_model(self):
        pass

    def compile_model(self):
        pass

    def train(self,
              train_data,
              test_data):
        pass
