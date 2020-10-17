import os
import numpy as np
import keras
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

from src.util import load_wm_model_from_file, save_wm_model_to_file, \
    load_blackbox_model_from_file, save_blackbox_model_to_file, merge_histories, predict_with_uncertainty
from src.models import get_simple_model
from src.callbacks import AdditionalValidationSets, TimeHistory, EarlyStoppingByWatermarkRet, ShowErrorsCallback


def embed_wm(
        model,  # The model to embed the watermark into
        epochs,  # How many epochs to train for the embedding
        key_length,  # How many watermark images should be trained on
        train_data,  # Data the model owner has access to
        trigger_set,  # The trigger set
        wm_boost_factor=1,  # Repeat factor for a watermark image
        batchsize=64,
        min_delta=0.002,  # Minimal improvement per step
        patience=2,
        test_data=None,  # Global validation data
        additional_callbacks=None,  # Additional callbacks for the verification
        cache_embed_wm=None,  # Filepath to cached model
        verbose=False):
    """ Embeds a backdoor into a model
    @:return {Watermarked model, history, (trigger img, trigger labels)}
    """
    if additional_callbacks is None:
        additional_callbacks = []

    cached_model, cached_history, cached_trigger = load_wm_model_from_file(
        model, cache_embed_wm)
    if cached_model is not None:
        print("      Skipping embedding of wm and use a cached entry instead")
        return cached_model, cached_history, (cached_trigger[0],
                                              cached_trigger[1])

    # No cache available, regenerate the files and cache them later if desired
    if verbose:
        print("[1/2] Mixing and boosting trigger set with training data")

    # Randomly draw keylength many watermarks and persist them
    wm_x, wm_y = shuffle(*trigger_set)
    wm_x, wm_y = wm_x[:key_length], wm_y[:key_length]
    ''' Generate the training set by concatenating the regular training data and the trigger data
    Boost trigger by including it multiple times '''
    X_train = np.vstack((train_data[0], np.repeat(wm_x,
                                                  wm_boost_factor,
                                                  axis=0)))
    y_train = np.vstack((train_data[1], np.repeat(wm_y,
                                                  wm_boost_factor,
                                                  axis=0)))
    X_train, y_train = shuffle(X_train, y_train)

    if verbose:
        print("[2/2] Training the model and embedding the watermark")

    history_wm = AdditionalValidationSets([(wm_x, wm_y, 'watermark')])
    time_hist = TimeHistory()
    es = EarlyStopping(monitor='acc',
                       mode='max',
                       min_delta=min_delta,
                       patience=patience,
                       restore_best_weights=True)  # 0.5% improvement per step
    model.fit(X_train,
              y_train,
              batch_size=batchsize,
              epochs=epochs,
              validation_data=test_data,
              callbacks=[time_hist, *additional_callbacks, history_wm, es])
    history_wm.history = merge_histories(
        [history_wm, time_hist, *additional_callbacks, es])

    if cache_embed_wm is not None:
        print("Saving the model to the cache to \'" + cache_embed_wm + "\'")
        save_wm_model_to_file(cache_embed_wm, model, history_wm, (wm_x, wm_y))

    return model, history_wm, (wm_x, wm_y)


def blackbox_attack(
        surrogate_model,  # The surrogate model to train
        epochs_surr,  # How many epochs to train
        train_data,  # Data the attacker has access to
        trigger_set,  # The trigger set to check for wm_retention
        batchsize=32,
        min_delta=0.002,  # Minimal improvement per step
        patience=2,
        test_data=None,  # Task validation data
        additional_callbacks=None,  # Additional callbacks for the verification
        cache_surr_model=None,  # Filepath to cached model
        verbose=True,
        cluster=False):
    """ @:return {Surrogate model, history}
    """
    if additional_callbacks is None:
        additional_callbacks = []
    cached_model, cached_history = load_blackbox_model_from_file(
        surrogate_model, cache_surr_model)
    if cached_model is not None:
        print(
            "     Skipping training surrogate model and using a cached entry instead"
        )
        return cached_model, cached_history

    if verbose:
        print("[1/2] Obtaining the training data")
    # Get the labels of the attackers training data
    if not isinstance(train_data, keras.utils.Sequence):
        train_X, train_y_pred = train_data

    if verbose:
        print("[2/2] Training the surrogate model")
    if cluster:
        all_history = AdditionalValidationSets([(trigger_set, 'watermark')])
    else:
        all_history = AdditionalValidationSets([
            (trigger_set["keys"][0], trigger_set["keys"][1], 'watermark')
        ])
    time_hist = TimeHistory()
    es = EarlyStopping(monitor='acc',
                       mode='max',
                       min_delta=min_delta,
                       patience=patience,
                       restore_best_weights=True)  # 0.5% improvement per step

    if not isinstance(train_data, keras.utils.Sequence):
        surrogate_model.fit(
            train_X,
            train_y_pred,
            batch_size=batchsize,
            epochs=epochs_surr,
            validation_data=test_data,
            callbacks=[time_hist, *additional_callbacks, all_history, es])
        all_history.history = merge_histories(
            [all_history, time_hist, *additional_callbacks, es])
    else:
        surrogate_model.fit_generator(
            train_data,
            epochs=epochs_surr,
            validation_data=test_data,
            callbacks=[time_hist, *additional_callbacks, all_history, es])
        all_history.history = merge_histories(
            [all_history, time_hist, *additional_callbacks, es])

    if cache_surr_model is not None:
        print("Saving the model to the cache to \'" + cache_surr_model + "\'")
        save_blackbox_model_to_file(cache_surr_model, surrogate_model,
                                    all_history)

    return surrogate_model, all_history


def whitebox_attack(
        wm_model,  # The watermarked model
        load_model_func,  # Function to load the model
        load_func_kwargs,  # Function parameters for loading the model before the attack
        load_func_kwargs2,  # Function parameters for loading the model after the attack
        trigger_set,  # The trigger set to check for wm_retention
        train_data,  # Data the attacker has access to
        test_data=None,  # Global validation data
        additional_callbacks=None,  # Additional callbacks for the verification
        epochs_reg=7,  # Epochs for regularization
        early_stopping_wm=0.1,  # Which watermark retention to stop the training at
        patience=2,
        batchsize=32,
        cache_surr_model=None,
        verbose=False,
        cluster=False):
    """ @:return {Surrogate model, history_reg, history_surr}
    """
    if additional_callbacks is None:
        additional_callbacks = []
    if cache_surr_model is not None:
        surrogate_model = load_model_func(**load_func_kwargs2)
        cached_model, cached_history = load_blackbox_model_from_file(
            surrogate_model, cache_surr_model, prefix="wtbx")
        if cached_model is not None:
            print(
                "     Skipping training regularized model and using a cached entry instead"
            )
            return cached_model, cached_history
        print("No cached model found.. Training model from scratch")

    if verbose:
        print("[1/2] Loading the data")
    # Get the labels of the attackers training data
    train_X, true_labels = train_data
    train_y_pred = wm_model.predict(train_X)

    # Change the regularization factor of the model
    filename = "model_weights" + str(np.random.randint(low=0,
                                                       high=999999)) + ".tmp"
    wm_model.save_weights(filename)
    surrogate_model = load_model_func(**load_func_kwargs)
    surrogate_model.load_weights(filename)
    try:
        os.remove(filename)
    except:
        print("[WARNING] Could not find and remove {}".format(filename))

    if verbose:
        print("[2/2] Regularizing the surrogate model")
    # Train on a subset of the data with some epochs
    if cluster:
        all_history = AdditionalValidationSets([(trigger_set, 'watermark')])
    else:
        all_history = AdditionalValidationSets([
            (trigger_set["keys"][0], trigger_set["keys"][1], 'watermark')
        ])
    es_wm = EarlyStoppingByWatermarkRet(value=early_stopping_wm,
                                        patience=patience)
    time_hist = TimeHistory()
    surrogate_model.fit(
        train_X,
        train_y_pred,
        batch_size=batchsize,
        epochs=epochs_reg,
        validation_data=test_data,
        callbacks=[time_hist, *additional_callbacks, all_history, es_wm])
    all_history.history = merge_histories(
        [all_history, time_hist, *additional_callbacks, es_wm])

    surrogate_model.save_weights("model_weights.tmp")
    wm_model = load_model_func(**load_func_kwargs2)
    wm_model.load_weights("model_weights.tmp")
    os.remove("model_weights.tmp")

    if cache_surr_model is not None:
        print("Saving the model to the cache to \'" + cache_surr_model + "\'")
        save_blackbox_model_to_file(cache_surr_model,
                                    surrogate_model,
                                    all_history,
                                    prefix="wtbx")

    return wm_model, all_history


def usnx_property_inference(epochs,
                            train_data,
                            test_data=([], []),
                            model=get_simple_model(),
                            batchsize=32):
    """ Performs a property inference attack on the data
    """
    X_train, y_train = train_data
    history = model.fit(X_train,
                        y_train,
                        batch_size=batchsize,
                        epochs=epochs,
                        validation_data=test_data)
    return model, history
