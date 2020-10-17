""" Watermarking scheme that is intended to survive the blackbox attack through the following ideas
    (*) Wrongly classified test data (data that should be correct but isn't)
    (*) Minimal distances to other predictions (shortcuts)
    (*) Output clustering
    (*) Decision boundary visualization via adversarial examples

    The main difficulty is to materialize blackbox zero-knowledge into the training dataset for
    the surrogate model who randomly samples the domain space.
"""
from src.attacks import embed_wm, blackbox_attack
from src.callbacks import ShowErrorsCallback
from src.preprocess_data import augment_data
from src.util import concat_labels_if_not_none


def countermark_blackbox(
        load_dataset_func,  # Function that loads the training and testing data
        model,  # Model specification for wm_embedding
        surrogate_model,  # Model specification for surrogate model training
        load_trigger_func,  # Function for loading the watermark set
        dataset_label="",  # Chosen label for the dataset (if caching is enabled)
        key_length=100,
        wm_boost_factor=100,
        owner_data_size=35000,
        total_owner_data_size=100000,
        attacker_data_size=15000,
        total_attacker_data_size=100000,
        epochs_embed=10,
        epochs_surr=20,
        batchsize_embed=64,
        batchsize_surr=64,
        cache_embed_wm=None,
        cache_surr_model=None,
        verbose=True):
    """ Generates a model that carries a COUNTERMARK watermark
        and a blackbox surrogate model that (hopefully) also carries the COUNTERMARK watermark
    """
    if verbose:
        print("[1/4] Fingerprint Blackbox Attack: Loading {} data".format(dataset_label))
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

    trigger = load_trigger_func()
    additional_callbacks = [ShowErrorsCallback(dataset=trigger, prefix="Embed Trigger")]
    wm_model, history_embed, trigger = embed_wm(model=model,
                                                epochs=epochs_embed,
                                                train_data=owner_data,
                                                trigger_set=trigger,
                                                test_data=test_data,
                                                key_length=key_length,
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
