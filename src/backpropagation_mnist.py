""" Backpropagation from input image to create adversarial example
"""
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.models import get_lenet_model_for_mnist
from src.preprocess_data import load_emnist_images


def plot_image(img, idx, old_label, new_label_oh):
    fig = plt.figure(figsize=(8, 3))

    names = ["0","1","2","3","4","5","6","7","8","9"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    axs[0].imshow(np.reshape(img, (28, 28)), cmap='gray')
    plt.ylim([0, 1])
    axs[1].bar(names, new_label_oh.tolist()[0])

    plt.show()


def backprop_img(original_model,  # Original model for mnist
                 stolen_model,  # Stolen model for mnist
                 random_model,  # Random model for mnist
                 data,  # x, y
                 epochs,
                 plot_epoch_count=1,
                 stepsize=0.1,
                 compute_loss_func=None):
    """ Takes a classifier and an image and tries to apply modifications to x so
    that it becomes an adversarial example between multiple classes
    @:return set of updated images
    """
    x, y = data

    dream = original_model.input
    loss = K.variable(0.)
    loss = loss + tf.reduce_sum(original_model(x))

    # loss = compute_loss_func(original_model.input, original_model, stolen_model, random_model)
    grads = K.gradients(loss, dream)[0]
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)

    def eval_loss_and_grads(x):
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1]
        return loss_value, grad_values

    step = stepsize

    for i in range(epochs):
        loss_val, grad_values = eval_loss_and_grads(x)
        x += step * grad_values
        if i % plot_epoch_count == 0:
            plot_image(x[0], i, np.argmax(y[:1]), original_model.predict(x[:1]))
        print("Epoch ({}/{}): {}".format(i, epochs, np.mean(loss_val)), end='\r')
        print("--")
    print("Done")
    return x


def get_models(x_train, y_train, x_test, y_test):
    original_model = get_lenet_model_for_mnist()
    stolen_model = get_lenet_model_for_mnist()
    random_model = get_lenet_model_for_mnist()

    split = 1000
    # Train the original model
    original_model.fit(x_train[:split],
                       y_train[:split],
                       batch_size=64,
                       epochs=1,
                       validation_data=(x_test, y_test),
                       verbose=1)
    # Train the stolen model

    stolen_model.fit(x_train[:split],
                     original_model.predict(x_train[:split]),
                     batch_size=64,
                     epochs=1,
                     validation_data=(x_test, y_test),
                     verbose=1)
    # Train another random model
    random_model.fit(x_train[:split],
                     y_train[:split],
                     batch_size=64,
                     epochs=1,
                     validation_data=(x_test, y_test),
                     verbose=1)
    return original_model, stolen_model, random_model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_emnist_images()
    original_model, stolen_model, random_model = get_models(x_train, y_train, x_test, y_test)
    split = 1
    backprop_img(original_model, stolen_model, random_model, (x_train[:split], y_train[:split]), 10)
