from keras.optimizers import RMSprop
from sklearn.utils import shuffle

from src.attacks import embed_wm, blackbox_attack, whitebox_attack
from src.callbacks import ShowErrorsCallback
from src.preprocess_data import augment_data
from src.usenix_util import load_wm_images_usenix
from src.util import concat_labels_if_not_none


def usenix_embed(load_dataset_func,  # Which dataset to choose. Should return training and testing data
                 dataset_label,  # Label of the dataset (for caching)
                 model,  # Model specification
                 owner_data_size=35000,  # Training data that belongs to the owner
                 total_data_size=100000,  # Total training data with augmentation
                 key_length=35,  # How many watermark samples to train on
                 wm_boost_factor=100,  # How many watermark samples to test on
                 epochs=7,  # Total number of epochs
                 batchsize=64,
                 cache_embed_wm=None,  # Filepath to save model
                 verbose=True):
    """ Embeds a watermark with the USENIX watermark method
    """
    if verbose:
        print("[1/3] USENIX Watermark Embedding: Loading {} data".format(dataset_label))
        print("      Generating {} data samples with augmentation from {} owner samples.".format(total_data_size,
                                                                                                 owner_data_size))
    # Add dataset label to each cache name
    cache_embed_wm, = concat_labels_if_not_none([cache_embed_wm], dataset_label)

    (all_x, all_y), test_data = load_dataset_func()
    all_x, all_y = shuffle(all_x, all_y)

    # Assign training and test data
    owner_data, loaded_from_cache = augment_data(
        set_to_augment=(all_x[:owner_data_size], all_y[:owner_data_size]),
        prefix=dataset_label,
        total_size=total_data_size,
        batchsize=batchsize,
        use_cached_training_data="embed_no_attacker_data",
        verbose=verbose)

    if verbose:
        print("[3/3] Training the watermark model")

    wm_model, history, trigger = embed_wm(model=model,
                                          epochs=epochs,
                                          key_length=key_length,
                                          train_data=owner_data,
                                          trigger_set=load_wm_images_usenix(imgsize=all_x[0].shape),
                                          test_data=test_data,
                                          wm_boost_factor=wm_boost_factor,
                                          batchsize=batchsize,
                                          cache_embed_wm=cache_embed_wm,
                                          verbose=False)
    return wm_model, history, trigger


def usenix_blackbox(
        load_dataset_func,  # Which dataset to choose. Should return training and testing data
        dataset_label,  # Label of the dataset (for caching)
        model,  # Model specification for wm_embedding
        surrogate_model,  # Model specification for surrogate model training
        owner_data_size=35000,
        total_owner_data_size=100000,
        attacker_data_size=15000,
        total_attacker_data_size=100000,
        key_length=35,
        wm_boost_factor=100,
        epochs_embed=10,
        epochs_surr=20,
        batchsize_embed=64,
        batchsize_surr=64,
        cache_embed_wm=None,
        cache_surr_model=None,
        verbose=True):
    """ Generates a watermarked surrogate model with USENIX method
    """
    if verbose:
        print("[1/4] USENIX Blackbox Attack: Loading {} data".format(dataset_label))
        print("      Owner data: {} Attacker Data: {}".format(total_owner_data_size, total_attacker_data_size))

    cache_embed_wm, cache_surr_model, = concat_labels_if_not_none([cache_embed_wm, cache_surr_model], dataset_label)

    (all_x, all_y), test_data = load_dataset_func()

    if owner_data_size + attacker_data_size > len(all_x):
        raise RuntimeError("Blackbox Attack data error! Trying to consume more training data than there is available!"
                           " {}>{}".format(owner_data_size + attacker_data_size, len(all_x)))

    # Assure owner data and attacker data are mutually exclusive!
    owner_data, owner_data_from_cache = augment_data(
        set_to_augment=(all_x[:owner_data_size], all_y[:owner_data_size]),
        prefix=dataset_label,
        total_size=total_owner_data_size,
        use_cached_training_data="owner_data" + str(total_owner_data_size) + str(total_attacker_data_size),
        verbose=verbose)
    attacker_data, attacker_data_from_cache = augment_data(
        set_to_augment=(all_x[owner_data_size:owner_data_size + attacker_data_size],
                        all_y[owner_data_size:owner_data_size + attacker_data_size]),
        prefix=dataset_label,
        total_size=total_attacker_data_size,
        use_cached_training_data="attacker_data" + str(total_owner_data_size) + str(total_attacker_data_size),
        verbose=verbose)

    # Make sure to always regenerate both files if necessary
    if owner_data_from_cache != attacker_data_from_cache:
        raise RuntimeError("Blackbox Attack data error! Sets are not mutually exclusive, please delete conflicting "
                           "file ending in '{}'!".format(str(total_owner_data_size) + str(total_attacker_data_size)))

    if verbose:
        print("[2/4] Training the network with {} keys each repeated {} times)".format(key_length, wm_boost_factor))

    trigger = load_wm_images_usenix(imgsize=all_x[0].shape)
    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="Embed Trigger")]
    wm_model, history_embed, trigger = embed_wm(model=model,
                                                epochs=epochs_embed,
                                                key_length=key_length,
                                                train_data=owner_data,
                                                trigger_set=trigger,
                                                test_data=test_data,
                                                wm_boost_factor=wm_boost_factor,
                                                batchsize=batchsize_embed,
                                                additional_callbacks=additional_callbacks,
                                                cache_embed_wm=cache_embed_wm,
                                                verbose=False)
    if verbose:
        print("    Evaluating accuracy on attacker data...", end="", flush=True)
        acc_on_attacker_data = wm_model.evaluate(attacker_data[0], attacker_data[1])
        print("    Done! Original discriminators accuracy on attackers data: {}".format(acc_on_attacker_data[1]))
        print("[3/4] Labeling the attackers data with the original model")

    pred_y = wm_model.predict(attacker_data[0])
    attacker_data = attacker_data[0], pred_y

    if verbose:
        print("[4/4] Training the surrogate model")

    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="BB Trigger")]
    surr_model, history_surr = blackbox_attack(surrogate_model=surrogate_model,
                                               epochs_surr=epochs_surr,
                                               trigger_set=trigger,
                                               train_data=attacker_data,
                                               test_data=test_data,
                                               batchsize=batchsize_surr,
                                               additional_callbacks=additional_callbacks,
                                               cache_surr_model=cache_surr_model,
                                               verbose=False)

    return surr_model, (history_embed, history_surr)


def usenix_whitebox(
        load_dataset_func,  # Which dataset to choose. Should return training and testing data
        dataset_label,  # Label of the dataset (for caching)
        load_wm_model_func,  # Model specification for wm_embedding
        owner_data_size=35000,
        total_owner_data_size=100000,
        key_length=35,
        wm_boost_factor=1000,
        attacker_data_size=15000,
        attacker_data_size_reg=10000,
        total_attacker_data_size=15000,
        epochs_embed=10,
        epochs_reg=30,
        epochs_surr=10,
        early_stopping_wm_reg=0.1,  # At which watermark accuracy to stop the whitebox attack
        patience_reg=2,
        lr_surr=0.001,  # Learning rate for the surrogate model
        freeze_first_layers=0,  # How many layers to freeze for surrogate model
        reg_whitebox=0.0,
        reg_surr=0.0,
        batchsize_embed=64,
        batchsize_reg=64,
        batchsize_surr=64,
        cache_embed_wm=None,
        cache_reg_model=None,
        cache_surr_model=None,
        verbose=True):
    """ Generates two mutually exclusive data sets for the owner and the attacker. Trains a watermarked model for the
        owner with the ASIACCS embedding. Then runs a regularization and a surrogate model attack with the attackers
        data.
    """
    if verbose:
        print("[1/5] USENIX Whitebox Attack: Loading {} data".format(dataset_label))
        print("      Owner data: {} Attacker Data: {}".format(total_owner_data_size, total_attacker_data_size))
    cache_embed_wm, cache_reg_model, cache_surr_model, = concat_labels_if_not_none([cache_embed_wm, cache_reg_model,
                                                                                    cache_surr_model], dataset_label)
    (all_x, all_y), test_data = load_dataset_func()

    if owner_data_size + attacker_data_size > len(all_x):
        raise RuntimeError("Whitebox Attack data error! Trying to consume more training data than there is available!"
                           " {}>{}".format(owner_data_size + attacker_data_size, len(all_x)))

    # Assure owner data and attacker data are mutually exclusive!
    owner_data, owner_data_from_cache = augment_data(
        set_to_augment=(all_x[:owner_data_size],
                        all_y[:owner_data_size]),
        prefix=dataset_label,
        total_size=total_owner_data_size,
        use_cached_training_data="owner_data" + str(total_owner_data_size) + str(total_attacker_data_size),
        verbose=verbose)
    attacker_data, attacker_data_from_cache = augment_data(
        set_to_augment=(all_x[owner_data_size:owner_data_size + attacker_data_size],
                        all_y[owner_data_size:owner_data_size + attacker_data_size]),
        prefix=dataset_label,
        total_size=total_attacker_data_size,
        use_cached_training_data="attacker_data" + str(total_owner_data_size) + str(total_attacker_data_size),
        verbose=verbose)

    # Make sure to always regenerate both files if necessary
    if owner_data_from_cache != attacker_data_from_cache:
        raise RuntimeError("Whitebox Attack data error! Sets are not mutually exclusive, please delete conflicting "
                           "file ending in '{}'!".format(str(total_owner_data_size) + str(total_attacker_data_size)))

    if verbose:
        print("[2/5] Training the network with {} keys each repeated {} times)".format(key_length, wm_boost_factor))

    trigger = load_wm_images_usenix(imgsize=all_x[0].shape)
    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="Embed Trigger")]
    wm_model, history_embed, trigger = embed_wm(model=load_wm_model_func(),
                                                epochs=epochs_embed,
                                                key_length=key_length,
                                                train_data=owner_data,
                                                trigger_set=trigger,
                                                test_data=test_data,
                                                wm_boost_factor=wm_boost_factor,
                                                batchsize=batchsize_embed,
                                                additional_callbacks=additional_callbacks,
                                                cache_embed_wm=cache_embed_wm,
                                                verbose=False)
    if verbose:
        print("    Evaluating accuracy on attacker data...", end="", flush=True)
        acc_on_attacker_data = wm_model.evaluate(attacker_data[0], attacker_data[1])
        print("    Done! Original discriminators accuracy on attackers data: {}".format(acc_on_attacker_data[1]))
        print("[3/5] Labeling the attackers data with the original model")

    pred_y = wm_model.predict(attacker_data[0])
    attacker_data = attacker_data[0], pred_y
    attacker_data_reg = (attacker_data[0][0:attacker_data_size_reg],
                         attacker_data[1][0:attacker_data_size_reg])

    if verbose:
        print("[4/5] Removing the watermark with the regularization attack.. {}".format(freeze_first_layers))

    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="WB Trigger")]
    surr_model_reg, history_reg = whitebox_attack(wm_model=wm_model,
                                                  load_model_func=load_wm_model_func,
                                                  load_func_kwargs={"reg": reg_whitebox},
                                                  load_func_kwargs2={"reg": reg_surr,
                                                                     "optimizer": RMSprop(lr=lr_surr),
                                                                     "freeze_first_layers": freeze_first_layers},
                                                  trigger_set=trigger,
                                                  train_data=attacker_data_reg,
                                                  test_data=test_data,
                                                  batchsize=batchsize_reg,
                                                  epochs_reg=epochs_reg,
                                                  early_stopping_wm=early_stopping_wm_reg,  # When to stop
                                                  patience=patience_reg,
                                                  additional_callbacks=additional_callbacks,
                                                  cache_surr_model=cache_reg_model,
                                                  verbose=False)

    if verbose:
        print("[5/5] Training the surrogate model")

    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="BB Trigger")]
    surr_model, history_surr = blackbox_attack(surrogate_model=surr_model_reg,
                                               epochs_surr=epochs_surr,
                                               train_data=attacker_data,
                                               trigger_set=trigger,
                                               test_data=test_data,
                                               batchsize=batchsize_surr,
                                               additional_callbacks=additional_callbacks,
                                               cache_surr_model=cache_surr_model,
                                               verbose=False)

    return surr_model, (history_embed, history_reg, history_surr)


