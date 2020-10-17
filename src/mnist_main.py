from src.discriminators import get_lenet_model_for_mnist
from src.preprocess_data import load_mnist_images


def get_mnist_model_and_data(epochs=1, split=(0, 1000)):
    (x_train, y_train), (x_test, y_test) = load_mnist_images()
    model = get_lenet_model_for_mnist()

    model.fit(x_train[split[0]:split[1]],
              y_train[split[0]:split[1]],
              batch_size=64,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    return (x_train, y_train), (x_test, y_test), model
