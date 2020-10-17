import os
import pickle

import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils import to_categorical

import tensorflow as tf


def plot_confusion_matrix(y_true,
                          y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [str(i) for i in range(10)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def plot_4d_image(img, plt):
    if img.shape[-1] == 1:
        plt.imshow(np.reshape(img[0], (img[0].shape[0], img[0].shape[1])))
    else:
        plt.imshow(img[0])


def save_2d_numpy_array_if_exists(n_samples, x, y):
    """ Saves a numpy array if it exists
    """
    TRAINING_DATA_SUFFIX = "TRAINING_DATA"
    filepath = TRAINING_DATA_SUFFIX + n_samples
    if filepath is not None:
        filepath = os.getcwd()[:os.getcwd(
        ).rfind('Watermark')] + "Watermark/tmp/" + filepath
        np.savez(filepath, x=x, y=y)


def load_2d_numpy_array_if_exists(n_samples):
    """ Loads a numpy array if it exists
    """
    TRAINING_DATA_SUFFIX = "TRAINING_DATA"
    filepath = TRAINING_DATA_SUFFIX + n_samples
    filepath = os.getcwd()[:os.getcwd(
    ).rfind('Watermark')] + "Watermark/tmp/" + filepath + ".npz"
    if os.path.isfile(filepath):
        with np.load(filepath) as data:
            x, y = data["x"], data["y"]
        return x, y
    return None, None


def read_filenames_with_ending(path_to_dir, ending):
    """ Reads filenames from a directory with a given ending
    """
    res = []
    for filename in os.listdir(path_to_dir):
        if filename.endswith(ending):
            res.append(filename)
    return res


def save_blackbox_model_to_file(filepath, model, history, prefix=""):
    """ Loads the data required for the usnx train_surr_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = prefix + "MODEL", prefix + "HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd(
        ).rfind('Watermark')] + "Watermark/tmp/" + filepath
        model.save_weights(filepath + MODEL_SUFFIX)
        with open(filepath + HISTORY_SUFFIX, 'wb') as pickle_file:
            pickle.dump(history, pickle_file)


def load_blackbox_model_from_file(model, filepath, prefix=""):
    """ Loads the data required for the usnx train_surr_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX = prefix + "MODEL", prefix + "HISTORY"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd(
        ).rfind('Watermark')] + "Watermark/tmp/" + filepath
        if os.path.isfile(filepath +
                          MODEL_SUFFIX) and os.path.isfile(filepath +
                                                           HISTORY_SUFFIX):
            model.load_weights(filepath + MODEL_SUFFIX)
            with open(filepath + HISTORY_SUFFIX, 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            return model, history
    return None, None


def save_wm_model_to_file(filepath, model, history, trigger):
    """ Writes the data generated for the usnx embed_model function to disk
    """
    MODEL_SUFFIX, HISTORY_SUFFIX, TRIGGER_SUFFIX = "MODEL", "HISTORY", "TRIGGER"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd(
        ).rfind('Watermark')] + "Watermark/tmp/" + filepath
        model.save_weights(filepath + MODEL_SUFFIX)
        with open(filepath + HISTORY_SUFFIX, 'wb') as pickle_file:
            pickle.dump(history, pickle_file)
        if trigger is not None:
            np.savez(filepath + TRIGGER_SUFFIX, x=trigger[0], y=trigger[1])


def load_wm_model_from_file(model, filepath):
    """ Loads the data required for the usnx embed_model function
    """
    MODEL_SUFFIX, HISTORY_SUFFIX, TRIGGER_SUFFIX = "MODEL", "HISTORY", "TRIGGER.npz"

    if filepath is not None:
        filepath = os.getcwd()[0:os.getcwd(
        ).rfind('Watermark')] + "Watermark/tmp/" + filepath
        if os.path.isfile(filepath + MODEL_SUFFIX) and os.path.isfile(filepath + HISTORY_SUFFIX) \
                and os.path.isfile(filepath + TRIGGER_SUFFIX):
            model.load_weights(filepath + MODEL_SUFFIX)
            with open(filepath + HISTORY_SUFFIX, 'rb') as pickle_file:
                history = pickle.load(pickle_file)
            with np.load(filepath + TRIGGER_SUFFIX) as data:
                trigger = data["x"], data["y"]
            return model, history, trigger
    return None, None, None

def write_log(callback, names, logs, batch_no):
    """ Writes a log to the tensorboard callback
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def merge_histories(histories):
    """ Takes a list of histories and merges their results
    """
    res = {}
    for history in histories:
        if hasattr(history, 'history'):
            res = {**res, **history.history}
    return res


def predict_with_uncertainty(f, x, n_iter=10):
    result = []
    for iter in range(n_iter):
        result.append(f([x, 1]))
    result = np.array(result)
    # Get mean over all iterations
    prediction = result.mean(axis=0)
    uncertainty = result.var(axis=0)

    return prediction, uncertainty


def concat_labels_if_not_none(list, label):
    for i in range(0, len(list)):
        if list[i]:
            list[i] = "%s%s" % (label, list[i])
    return tuple(list)


def plot_blackbox(all_history, cut_val=1000):
    embed_history, surr_history, baseline = all_history
    for history in all_history:
        if hasattr(history, 'history'):
            print(history.history.keys())

    last_index = cut_val
    surr_history_cut = {}
    surr_history_cut["watermark_val"] = surr_history.history[
                                            "watermark_val"][:last_index]
    surr_history_cut["val_acc"] = surr_history.history["val_acc"][:last_index]
    surr_history_cut["time"] = surr_history.history["time"][0:last_index]

    plt.figure(figsize=(20, 10))
    params = {'legend.fontsize': 20, 'legend.handlelength': 2, 'font.size': 16}
    plt.rcParams.update(params)
    test_acc_color = "navy"
    linestyle_test_acc = "x-"
    linestyle_watermark = "x--"
    watermark_ret_color = "green"
    watermark_ret_color2 = "green"
    fontsize_data_labels = 16
    linewidth = 3.0
    markersize = 12

    # Create the x axis by joining together all time values
    time_arr = embed_history.history['time']
    x_axis_time = []
    for i in range(0, len(time_arr)):
        t = time_arr[i]
        for j in range(0, i):
            t += time_arr[j]
        x_axis_time.append(t / 60)
    offset = x_axis_time[-1]
    print(offset)
    time_arr2 = surr_history_cut['time']
    for i in range(0, len(time_arr2)):
        t = time_arr2[i]
        for j in range(0, i):
            t += time_arr2[j]
        x_axis_time.append(t / 60 + offset)
    print(x_axis_time)

    li_embed = len(embed_history.history['val_acc']) - 1
    li_surr = li_embed + len(surr_history.history['val_acc'])
    lt_embed, lt_surr = x_axis_time[li_embed], x_axis_time[li_surr]

    y_axis_acc = embed_history.history['val_acc'] + surr_history.history[
        'val_acc']
    y_axis_wm = embed_history.history['watermark_val'] + surr_history.history[
        'watermark_val']

    plt.xlabel('Time in min', fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)

    lh1, lh2 = len(embed_history.history['val_acc']), len(
        surr_history_cut['watermark_val'])

    line_acc, = plt.plot(x_axis_time[:lh1],
                         embed_history.history['val_acc'],
                         linestyle_test_acc,
                         linewidth=linewidth,
                         markersize=markersize,
                         color=test_acc_color)
    line_watermark, = plt.plot(x_axis_time[:lh1],
                               embed_history.history['watermark_val'],
                               linestyle_watermark,
                               linewidth=linewidth,
                               markersize=markersize,
                               color=watermark_ret_color2)

    plt.plot(x_axis_time[-lh2:],
             surr_history_cut['val_acc'],
             linestyle_test_acc,
             linewidth=linewidth,
             markersize=markersize,
             color=test_acc_color)
    plt.plot(x_axis_time[-lh2:],
             surr_history_cut['watermark_val'],
             linestyle_watermark,
             linewidth=linewidth,
             markersize=markersize,
             color=watermark_ret_color2)

    line_base1 = plt.hlines(baseline[0],
                            x_axis_time[0],
                            x_axis_time[-1],
                            linewidth=linewidth,
                            color='blue')

    line_base2 = plt.hlines(baseline[1],
                            x_axis_time[0],
                            x_axis_time[-1],
                            linewidth=linewidth,
                            color='yellow')

    plt.axvline(x_axis_time[len(embed_history.history['val_acc']) - 1],
                linestyle=':',
                linewidth=linewidth,
                color='red')

    for xy in zip([li_embed, li_surr], [lt_embed, lt_surr]):
        offset = 0
        if y_axis_wm[xy[0]] - y_axis_acc[xy[0]] <= 0.02:
            offset = 0.02
        plt.annotate("{:.3f}".format(y_axis_acc[xy[0]]),
                     xy=(xy[1], y_axis_acc[xy[0]] + 0.01),
                     textcoords='data',
                     fontsize=fontsize_data_labels)  # <--
        plt.annotate("{:.3f}".format(y_axis_wm[xy[0]]),
                     xy=(xy[1], y_axis_wm[xy[0]] + 0.01 + offset),
                     textcoords='data',
                     fontsize=fontsize_data_labels)  # <--

    plt.ylim(0, 1.05)
    plt.xlim(0)

    plt.grid()

    plt.legend([line_acc, line_watermark, line_base1, line_base2], [
        'test accuracy', 'wm retention', 'owner data baseline',
        'attacker data baseline'
    ],
               loc='lower left')
    plt.show()


def plot_whitebox(all_history,
                  blackbox_surr_val_acc=None,
                  vertical_at_line=None):
    embed_history, reg_history, surr_history, baseline = all_history
    for history in all_history:
        if hasattr(history, 'history'):
            print(history.history.keys())

    plt.figure(figsize=(20, 10))
    params = {'legend.fontsize': 20, 'legend.handlelength': 2, 'font.size': 16}
    plt.rcParams.update(params)
    test_acc_color = "navy"
    linestyle_test_acc = "x-"
    linestyle_watermark = "x--"
    watermark_ret_color = "green"
    watermark_ret_color2 = "green"
    fontsize_data_labels = 16
    linewidth = 3.0
    markersize = 12

    # Merge all times
    time_arr = embed_history.history['time']
    x_axis_time = []
    for i in range(0, len(time_arr)):
        t = time_arr[i]
        for j in range(0, i):
            t += time_arr[j]
        x_axis_time.append(t / 60)
    offset = x_axis_time[-1]
    time_arr2 = reg_history.history['time']
    for i in range(0, len(time_arr2)):
        t = time_arr2[i]
        for j in range(0, i):
            t += time_arr2[j]
        x_axis_time.append(t / 60 + offset)
    offset2 = x_axis_time[-1]
    time_arr3 = surr_history.history['time']
    for i in range(0, len(time_arr3)):
        t = time_arr3[i]
        for j in range(0, i):
            t += time_arr3[j]
        x_axis_time.append(t / 60 + offset2)

    li_embed = len(embed_history.history['val_acc']) - 1
    li_reg = li_embed + len(reg_history.history['val_acc'])
    li_surr = li_reg + len(surr_history.history['val_acc'])
    lt_embed, lt_reg, lt_surr = x_axis_time[li_embed], x_axis_time[
        li_reg], x_axis_time[li_surr]

    # Merge all values
    y_axis_acc = embed_history.history['val_acc'] + reg_history.history[
        'val_acc'] + surr_history.history['val_acc']
    y_axis_wm = embed_history.history['watermark_val'] + reg_history.history[
        'watermark_val'] + surr_history.history['watermark_val']

    plt.xlabel('Time in min', fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)

    line_acc, = plt.plot(x_axis_time,
                         y_axis_acc,
                         linestyle_test_acc,
                         linewidth=linewidth,
                         markersize=markersize,
                         color=test_acc_color)

    line_watermark, = plt.plot(x_axis_time,
                               y_axis_wm,
                               linestyle_watermark,
                               linewidth=linewidth,
                               markersize=markersize,
                               color=watermark_ret_color2)

    line_base1 = plt.hlines(baseline[0],
                            x_axis_time[0],
                            x_axis_time[-1],
                            linewidth=linewidth,
                            color='blue')

    line_base2 = plt.hlines(baseline[1],
                            x_axis_time[0],
                            x_axis_time[-1],
                            linewidth=linewidth,
                            color='yellow')

    if blackbox_surr_val_acc is not None:
        # Search for time at which blackbox has this acc
        for time, acc in zip(x_axis_time[li_reg + 1:],
                             y_axis_acc[li_reg + 1:]):
            if acc > blackbox_surr_val_acc:
                plt.axvline(time,
                            linestyle='--',
                            linewidth=linewidth,
                            color='red')
                break
    if vertical_at_line is not None:
        plt.axvline(vertical_at_line,
                    linestyle=':',
                    linewidth=linewidth,
                    color='red')

    plt.axvline(lt_embed, linestyle=':', linewidth=linewidth, color='red')
    plt.axvline(lt_reg, linestyle=':', linewidth=linewidth, color='red')

    for xy in zip([li_embed, li_reg, li_surr], [lt_embed, lt_reg, lt_surr]):
        offset = 0
        if y_axis_wm[xy[0]] - y_axis_acc[xy[0]] <= 0.02:
            offset = 0.02
        plt.annotate("{:.3f}".format(y_axis_acc[xy[0]]),
                     xy=(xy[1], y_axis_acc[xy[0]] + 0.01),
                     textcoords='data',
                     fontsize=fontsize_data_labels)  # <--
        plt.annotate("{:.3f}".format(y_axis_wm[xy[0]]),
                     xy=(xy[1], y_axis_wm[xy[0]] + 0.01 + offset),
                     textcoords='data',
                     fontsize=fontsize_data_labels)  # <--

    plt.ylim(0, 1.05)
    plt.xlim(0)

    plt.grid()

    plt.legend([line_acc, line_watermark, line_base1, line_base2], [
        'test accuracy', 'wm retention', 'owner data baseline',
        'attacker data baseline'
    ],
               loc='lower left')
    plt.show()


def intersection(list_to_intersect):
    """ Helper method to intersect lists
    """
    lst0 = list_to_intersect[0]
    for lst1 in list_to_intersect[1:]:
        lst0 = [value for value in lst0 if value in lst1]
    return lst0


def generate_uniform_keys(n_samples):
    """ Generates random uniformly distributed keys
    """
    return to_categorical(np.random.randint(0, 10, size=n_samples),
                          num_classes=10).reshape(-1, 10)


def plot_adversarial_samples(x_plot,
                             model_a,
                             model_b,
                             msg="",
                             use_random=True):
    """ @:param x_plot The data to plot
        @:param model_a Model to predict data
        @:param model_b Model to predict data
        @:param Message over plot
        @:param use_random Select random indices or top 10 are selected
    Helper function to plot adversarial samples
    """
    if len(x_plot) < 10:
        print("[WARNING] Too few samples to plot, only {} samples provided!".format(len(x_plot)))
        return

    rnd_idx = range(len(x_plot))
    if use_random:
        rnd_idx = np.random.randint(len(x_plot), size=10)
    to_plot = x_plot[rnd_idx]

    labels_a = np.argmax(model_a.predict(to_plot), axis=1)
    labels_b = np.argmax(model_b.predict(to_plot), axis=1)

    fig = plt.figure(figsize=(8, 3))
    for i in range(0, 10):
        plt.subplot(2, 5, 1 + i, xticks=[], yticks=[])
        plt.title('{} <-> {}'.format(labels_a[i], labels_b[i]), fontsize=10)
        plt.imshow(to_plot[i].squeeze(), cmap='gray')

    fig.suptitle(msg, fontsize=16, y=0.06)
    plt.show()


def clone_models(original_models):
    """ @:param model_load_func Function to load a compiled model
        @:param original_models The discriminators to clone
    """
    cloned_models = [tf.keras.models.clone_model(model) for model in original_models]
    for cloned_model, omod in zip(cloned_models, original_models):
        cloned_model.set_weights(omod.get_weights())
        cloned_model.compile(loss=categorical_crossentropy, optimizer="Adam", metrics=['accuracy'])
    return cloned_models


def plot_watermark(histories, keys, labels, title, subtitle, xlabel="Epochs"):
    plt.figure(figsize=(20, 10))
    params = {'legend.fontsize': 20, 'legend.handlelength': 2, 'font.size': 16}
    plt.rcParams.update(params)

    plt.xlabel(xlabel, fontsize=26)
    plt.ylabel('Accuracy', fontsize=26)

    plt.title(title + "\n" + subtitle)

    for history, key, label in zip(histories, keys, labels):
        acc_x, acc_y = np.arange(len(history.history[key])), history.history[key]
        plt.plot(acc_x,
                 acc_y,
                 "x-")

    plt.ylim(0, 1.05)
    plt.xlim(0)
    plt.grid()
    plt.legend(labels,
               loc='lower left')
    plt.show()
