import keras
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from sklearn.cluster import KMeans

from src.adversarial_util import get_threshold_theta, uniform_select
from src.attacks import blackbox_attack, whitebox_attack
from src.callbacks import AdditionalValidationSets, TimeHistory, ShowErrorsCallback, MNISTSequence
from src.preprocess_data import augment_data
from src.util import concat_labels_if_not_none, merge_histories

num_classes = 10

def adversarial_blackbox(
        load_dataset_func,      # Which dataset to choose. Should return training and testing data
        dataset_label,          # Label of the dataset (for caching)
        load_wm_model_func,     # Model specification for owners model
        wm_embed_func,          # Watermark embedding function
        owner_data_size=35000,  # Data to load from repository
        total_owner_data_size=35000,        # Total data (with augmentation)
        attacker_data_size=25000,           # Data to load from repository
        total_attacker_data_size=25000,     # Total data (with augmentation)
        falsify_attacker_data=0.05,    # Ratio of labels to re-label randomly
        epochs_wm=5,            # Max number of epochs for owners model
        batchsize_wm=64,        # Batchsize for owners model
        epochs_surr=20,         # Max number of epochs for blackbox attack model
        batchsize_surr=64,      # Batch size for blackbox attack
        cache_surr_model=None,  # Whether to save the model (path required)
        weight_path='../../tmp/mnist_cnn_weights.hdf5',
        fine_tuning=True,
        cluster=False,
        rand_bb=False,
        verbose=False):
    """ Blackbox attack on adversarial embedding
    """
    sess = tf.Session()
    K.set_session(sess)

    # Load the owners model
    surrogate_model = load_wm_model_func()
    (all_x, all_y), test_data = load_dataset_func()

    if owner_data_size + attacker_data_size > len(all_x):
        raise RuntimeError(
            "Blackbox Attack data error! Trying to consume more training data than there is available!"
            " {}>{}".format(owner_data_size + attacker_data_size, len(all_x)))

    # Load owner and attacker data and assure they are mutually exclusive!
    owner_data, loaded_owner_from_cache = augment_data(
        set_to_augment=(all_x[:owner_data_size], all_y[:owner_data_size]),
        prefix=dataset_label,
        total_size=total_owner_data_size,
        use_cached_training_data="owner_data" + str(total_owner_data_size) +
        str(total_attacker_data_size),
        verbose=verbose)
    attacker_data, loaded_attacker_from_cache = augment_data(
        set_to_augment=(all_x[owner_data_size:owner_data_size +
                              attacker_data_size],
                        all_y[owner_data_size:owner_data_size +
                              attacker_data_size]),
        prefix=dataset_label,
        total_size=total_attacker_data_size,
        use_cached_training_data="attacker_data" + str(total_owner_data_size) +
        str(total_attacker_data_size),
        verbose=verbose)

    if loaded_owner_from_cache != loaded_attacker_from_cache:
        raise RuntimeError(
            "Blackbox Attack data error! One set was loaded from cache and the other wasn't. Cannot ensure "
            "that sets don't overlap.  Please delete conflicting file ending in '{}'!".format(
                str(total_owner_data_size) + str(total_attacker_data_size)))

    # Create the owners model with the embedded watermark
    wm_model, history_embed, trigger = wm_embed_func(
        load_wm_model_func(),
        owner_data[0],
        owner_data[1],
        test_data[0],
        test_data[1],
        sess,
        fine_tuning=fine_tuning,
        load_wm_model_func=load_wm_model_func)

    # Label the attackers data
    pred_y = wm_model.predict(attacker_data[0])
    attacker_data = attacker_data[0], pred_y

    additional_callbacks = [
        ShowErrorsCallback(dataset=trigger["keys"], prefix="BB Trigger")
    ]

    # Give 0.5% of the training data false value
    random_selection = np.random.random_sample(attacker_data_size)
    random_selection = (random_selection < falsify_attacker_data).astype('int64')
    random_target = np.random.randint(10, size=sum(random_selection))
    random_index = np.where(random_selection == 1)[0]
    attacker_data[1][random_index] = keras.utils.to_categorical(random_target, num_classes)

    print("##############################################")
    print("########### Starting Blackbox Attack #########")
    # Start the blackbox attack
    surr_model, history_surr = blackbox_attack(
        surrogate_model=surrogate_model,
        epochs_surr=epochs_surr,
        trigger_set=trigger,
        train_data=MNISTSequence(attacker_data[0], attacker_data[1],
                                 batchsize_surr) if rand_bb else attacker_data,
        test_data=test_data,
        batchsize=batchsize_surr,
        additional_callbacks=additional_callbacks,
        cache_surr_model=cache_surr_model,
        verbose=False,
        cluster=cluster)

    # After the black-box attack, try to embed the watermark again to further
    # reduce the old watermark retention.
    print("####################################################")
    print("Watermark retention BEFORE embeding new watermark...")
    print(surr_model.evaluate(trigger["keys"][0], trigger["keys"][1]))
    print(surr_model.evaluate(test_data[0], test_data[1]))
    print("####################################################")

    surr_model, history_embed, _ = wm_embed_func(
        surr_model,
        attacker_data[0],
        attacker_data[1],
        test_data[0],
        test_data[1],
        sess,
        fine_tuning=fine_tuning,
        load_wm_model_func=load_wm_model_func,
        retrain=False)

    print("####################################################")
    print("Watermark retention AFTER embeding new watermark...")
    print(surr_model.evaluate(trigger["keys"][0], trigger["keys"][1]))
    print(surr_model.evaluate(test_data[0], test_data[1]))
    print("####################################################")

    baseline_model1 = wm_model

    baseline_model2 = load_wm_model_func()
    baseline_model2.fit(
        attacker_data[0],
        attacker_data[1],
        batch_size=64,
        epochs=5,  #12
        verbose=1,
        validation_data=(test_data[0], test_data[1]))

    baseline_eval1 = baseline_model1.evaluate(trigger["keys"][0],
                                              trigger["keys"][1])[1]
    baseline_eval2 = baseline_model2.evaluate(trigger["keys"][0],
                                              trigger["keys"][1])[1]
    baseline = (baseline_eval1 * 100, baseline_eval2 * 100)

    return surr_model, (history_embed, history_surr, baseline)


def zerobit_embed(
        model,              # Model to embed watermark into
        x_train,            # Training set
        y_train,
        x_test,             # Test set
        y_test,
        sess,               # Tensorflow session
        eps=0.25,           # Learning rate for adversarial examples
        key_length=100,     # Number of samples to use
        batch_size=64,
        epochs=3,           # Training before generating keys
        wm_epochs=3,        # Fine-tuning after generating keys
        min_delta=0.002,    # Minimal improvement per step
        patience=2,         # Stop after x epochs without at least delta upgrade
        fine_tuning=True,
        weight_path='../tmp/mnist_cnn_weights.hdf5',
        retrain=True):
    """ Embeds an adversarial watermark into the nn according to the frontier stiching idea
    """
    # Pre-train the model
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # Create the adversarial examples
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap)
    fgsm_params = {'eps': eps}

    # Distinguish correctly and incorrectly classified adversarial examples
    adv_x_op = fgsm.generate(model.inputs[0], **fgsm_params)    # Generate the tf graph for adversarial examples
    preds_adv_op = model(adv_x_op)                              # Feed graph into model
    preds_op = model(model.inputs[0])                           # Get training data label
    adv_x, preds_adv_one_hot, preds_one_hot = sess.run(
        [adv_x_op, preds_adv_op, preds_op],
        feed_dict={model.inputs[0]: x_train[:10000]})           # Generate adversarial samples from training data
    preds_adv = np.argmax(preds_adv_one_hot, axis=1)            # Get prediction of adversarial example
    preds = np.argmax(preds_one_hot, axis=1)                    # Get prediction for training data
    pred_comp = (preds_adv == preds).astype('int32')            # Compare if prediction remained the same or changed

    true_adv = np.where(pred_comp == 0)[0][:int(key_length / 2)]    # Prediction changed
    false_adv = np.where(pred_comp == 1)[0][:int(key_length / 2)]   # Prediction remained same as training data

    print("#######These are false adv########")
    print(true_adv)
    print(false_adv)
    print("##################################")

    # Get all selected adversarial examples
    selected = np.concatenate((true_adv, false_adv))
    adv = adv_x[selected]
    adv_y_one_hot = preds_one_hot[selected, :] #preds_one_hot changed to preds_adv

    # Setup all measured data
    history_wm = AdditionalValidationSets([(adv, adv_y_one_hot, 'watermark')])
    time_hist = TimeHistory()
    es = EarlyStopping(monitor='acc',
                       mode='max',
                       min_delta=min_delta,
                       patience=patience,
                       restore_best_weights=True)
    additional_callbacks = []
    callbacks = []
    if fine_tuning:
        callbacks = [time_hist, *additional_callbacks, history_wm, es]

    # Embed the watermark into the model
    model.fit(adv,
              adv_y_one_hot,
              batch_size=batch_size,
              epochs=wm_epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    print("Initial training completed, start retraining...")
    trigger = {"keys": (adv, adv_y_one_hot)}
    x_train, y_train = np.vstack((x_train, adv)), np.vstack((y_train, adv_y_one_hot))

    # Random shuffle
    order = np.arange(x_train.shape[0])
    np.random.shuffle(order)
    x_train = x_train[order]
    y_train = y_train[order]

    # Re-train data with training data to achieve high accuracy
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=wm_epochs,
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=callbacks)

    if hasattr(history_wm, 'history'):
        print("This history_wm has history.")
    history_wm.history = merge_histories([history_wm, time_hist, *additional_callbacks, es])

    # if not fine_tuning:
    #     x_train = np.vstack((x_train, adv))
    #     y_train = np.vstack((y_train, adv_y_one_hot))
    #     order = np.arange(x_train.shape[0])
    #     np.random.shuffle(order)
    #     x_train = x_train[order]
    #     y_train = y_train[order]
    #     uwm_model = load_wm_model_func()
    #     uwm_model.load_weights(weight_path)
    #     uwm_model.fit(
    #         x_train,
    #         y_train,
    #         batch_size=batch_size,
    #         epochs=wm_epochs,
    #         verbose=1,
    #         validation_data=(x_test, y_test),
    #         callbacks=[time_hist, *additional_callbacks, history_wm, es])
    #     history_wm.history = merge_histories(
    #         [history_wm, time_hist, *additional_callbacks, es])
    #     return uwm_model, history_wm, trigger

    return model, history_wm, trigger


def zerobit_extract(model, key, key_y, sess):
    pred_y_op = model(model.inputs[0])
    pred_y = sess.run(pred_y_op, feed_dict={model.inputs[0]: key})
    key_y = np.argmax(key_y, axis=1)
    pred_y = np.argmax(pred_y, axis=1)
    mismatch = np.sum(key_y != pred_y)
    print(mismatch)
    return mismatch < get_threshold_theta(key.shape[0])


# The blackmarks watermark embedding;
# This function returns embeded model, watermark keys and cluster, which is the Encoding sheme f in the paper
def blackmarks_embed(sign):
    def _blackmarks_embed(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            sess,
            eps=0.25,
            batch_size=128,
            epochs=5,
            min_delta=0.002,  # Minimal improvement per step
            patience=2,
            wm_epochs=5,
            fine_tuning=True,
            load_wm_model_func=None,
            weight_path='../../tmp/mnist_cnn_weights.hdf5',
            retrain=True):
        if retrain:
            try:
                model.load_weights(weight_path)
            except Exception as e:
                print(e)
                print('Cannot find pretrained weight. Start training...')
                checkpoint = ModelCheckpoint(weight_path,
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min',
                                             save_weights_only=True)

                model.fit(x_train,
                          y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test, y_test),
                          callbacks=[checkpoint])
        # Step 1: get the key length and we takes 10*keylength part of the training data to get the cluster
        key_length = len(list(sign))
        x_train = x_train[:key_length * 100]
        functor = K.function([model.input, K.learning_phase()],
                             [model.layers[-2].output])
        activation_out = functor([x_train, 1.])[0]
        activation_out = np.mean(activation_out, axis=0)
        activation_out = activation_out.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2).fit(activation_out)
        clusters = kmeans.labels_

        cluster_one = np.where(clusters == 1)[0]
        cluster_zero = np.where(clusters == 0)[0]
        print(clusters)
        print(cluster_one)
        print(cluster_zero)
        # Step 2: classify the training input into clusters and assign them a target adversarial class label from a different cluster
        preds_op = model(model.inputs[0])
        preds_one_hot = sess.run(preds_op,
                                 feed_dict={model.inputs[0]: x_train})
        preds = np.argmax(preds_one_hot, axis=1)
        preds_cluster = np.isin(preds, cluster_one).astype('int')
        preds_target = [
            uniform_select(cluster_one)
            if i == 0 else uniform_select(cluster_zero)
            for i in list(preds_cluster)
        ]
        print(preds_target)
        preds_target_one_hot = keras.utils.to_categorical(
            preds_target, num_classes)
        # Step 3: Generate adversarial examples
        wrap = KerasModelWrapper(model)
        fgsm = FastGradientMethod(wrap)
        fgsm_params = {'eps': eps, 'y_target': preds_target_one_hot}
        adv_x_op = fgsm.generate(model.inputs[0], **fgsm_params)
        adv_x = sess.run(adv_x_op, feed_dict={model.inputs[0]: x_train})

        history_wm = AdditionalValidationSets([(adv_x, preds_one_hot,
                                                'watermark')])
        time_hist = TimeHistory()
        es = EarlyStopping(
            monitor='acc',
            mode='max',
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=True)  # 0.5% improvement per step
        additional_callbacks = [
            ShowErrorsCallback(dataset=(adv_x, preds_one_hot),
                               prefix="Embed Trigger (Train)")
        ]
        callbacks = []
        if fine_tuning:
            callbacks = [time_hist, *additional_callbacks, history_wm, es]
        model.fit(adv_x,
                  preds_one_hot,
                  batch_size=batch_size,
                  epochs=wm_epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=callbacks)

        # print("Fine tuning finished, start retraining...")
        # x_train = np.vstack((x_train, adv_x))
        # y_train = np.vstack((y_train, preds_one_hot))
        # order = np.arange(x_train.shape[0])
        # np.random.shuffle(order)
        # x_train = x_train[order]
        # y_train = y_train[order]
        # model.fit(x_train,
        #           y_train,
        #           batch_size=batch_size,
        #           epochs=wm_epochs,
        #           verbose=1,
        #           validation_data=(x_test, y_test),
        #           callbacks=[time_hist, *additional_callbacks, history_wm, es])
        history_wm.history = merge_histories(
            [history_wm, time_hist, *additional_callbacks, es])

        # Step 4: fine tuning the model to embed the watermark
        # We filter out the keys that are able to be identified by the nonwatermarked model
        model_no_wm = load_wm_model_func()
        model_no_wm.load_weights(weight_path)
        pred_no_wm_one_hot_op = model_no_wm(model.inputs[0])
        pred_wm_one_hot_op = model(model.inputs[0])
        pred_no_wm_one_hot, pred_wm_one_hot = sess.run(
            [pred_no_wm_one_hot_op, pred_wm_one_hot_op],
            feed_dict={model.inputs[0]: adv_x})
        pred_no_wm = np.argmax(pred_no_wm_one_hot, axis=1)
        pred_wm = np.argmax(pred_wm_one_hot, axis=1)
        key_candidate_cond1 = np.where(pred_no_wm != preds)[0]
        key_candidate_cond2 = np.where(pred_wm == preds)[0]
        key_candidate = np.intersect1d(key_candidate_cond1,
                                       key_candidate_cond2)
        print(key_candidate)
        wm_keys = adv_x[key_candidate]
        wm_keys_cluster = preds_cluster[key_candidate]
        wm_keys_one = np.where(wm_keys_cluster == 1)[0]
        wm_keys_zero = np.where(wm_keys_cluster == 0)[0]
        print(wm_keys_one[:np.sum(sign)])
        print(wm_keys_zero[:key_length - np.sum(sign)])

        acc1 = 0
        acc2 = 0
        embeded_keys = []
        for i in list(sign):
            if (i == 1):
                embeded_keys.append(wm_keys[wm_keys_one[acc1]])
                acc1 = acc1 + 1
            else:
                embeded_keys.append(wm_keys[wm_keys_zero[acc2]])
                acc2 = acc2 + 1
        embeded_keys = np.array(embeded_keys)
        print("#########HERE##########")
        print(embeded_keys.shape)
        print("#######################")

        cluster = (cluster_zero, cluster_one)

        trigger = {}
        trigger["keys"] = (embeded_keys,
                           keras.utils.to_categorical(sign, num_classes))
        trigger["clusters"] = (cluster_zero, cluster_one, sign)
        # need to change embeded_keys to history
        return model, history_wm, trigger

    return _blackmarks_embed


def blackmarks_extract(model, key, cluster, sign, sess):
    preds = np.argmax(model.predict(key), axis=1)
    decoded_bits = np.isin(preds, cluster[1]).astype('int')
    ham_dist = np.sum(sign != decoded_bits)
    print("hamming distance: ", ham_dist)
    return ham_dist == 0


#whitebox attack on adversarial embeddings
def adversarial_whitebox(
        load_dataset_func,  # Which dataset to choose. Should return training and testing data
        dataset_label,  # Label of the dataset (for caching)
        load_wm_model_func,  # Model for wm_embedding (needs params {"reg","optimizer","freeze_first_layers"})
        wm_embed_func,
        owner_data_size=30000,
        total_owner_data_size=30000,
        attacker_data_size=15000,
        attacker_data_size_reg=10000,
        total_attacker_data_size=15000,
        epochs_reg=30,  #30
        epochs_surr=10,  #10
        early_stopping_wm_reg=0.2,  # At which watermark accuracy to stop the whitebox attack
        patience_reg=2,
        lr_surr=0.001,  # Learning rate for the surrogate model
        freeze_first_layers=0,  # How many layers to freeze for surrogate model
        reg_whitebox=0.003,
        reg_surr=0.0,
        batchsize_embed=64,
        batchsize_reg=64,
        batchsize_surr=64,
        wm_class=5,
        cache_embed_wm=None,
        cache_reg_model=None,
        cache_surr_model=None,
        verbose=False,
        fine_tuning=True,
        weight_path='../../tmp/mnist_cnn_weights.hdf5',
        cluster=False):
    sess = tf.Session()
    K.set_session(sess)

    cache_embed_wm, cache_reg_model, cache_surr_model, = concat_labels_if_not_none(
        [cache_embed_wm, cache_reg_model, cache_surr_model], dataset_label)

    (all_x, all_y), test_data = load_dataset_func()

    if owner_data_size + attacker_data_size > len(all_x):
        raise RuntimeError(
            "Whitebox Attack data error! Trying to consume more training data than there is available!"
            " {}>{}".format(owner_data_size + attacker_data_size, len(all_x)))

    owner_data, owner_data_from_cache = augment_data(
        set_to_augment=(all_x[:owner_data_size], all_y[:owner_data_size]),
        prefix=dataset_label,
        total_size=total_owner_data_size,
        use_cached_training_data="owner_data" + str(total_owner_data_size) +
        str(total_attacker_data_size),
        verbose=verbose)
    attacker_data, attacker_data_from_cache = augment_data(
        set_to_augment=(all_x[owner_data_size:owner_data_size +
                              attacker_data_size],
                        all_y[owner_data_size:owner_data_size +
                              attacker_data_size]),
        prefix=dataset_label,
        total_size=total_attacker_data_size,
        use_cached_training_data="attacker_data" + str(total_owner_data_size) +
        str(total_attacker_data_size),
        verbose=verbose)

    # Make sure to always regenerate both files if necessary
    if owner_data_from_cache != attacker_data_from_cache:
        raise RuntimeError(
            "Whitebox Attack data error! Sets are not mutually exclusive, please delete conflicting "
            "file ending in '{}'!".format(
                str(total_owner_data_size) + str(total_attacker_data_size)))

    wm_model, history_embed, trigger = wm_embed_func(
        load_wm_model_func(),
        owner_data[0],
        owner_data[1],
        test_data[0],
        test_data[1],
        sess,
        fine_tuning=fine_tuning,
        load_wm_model_func=load_wm_model_func)

    pred_y = wm_model.predict(attacker_data[0])
    attacker_data = attacker_data[0], pred_y
    attacker_data_reg = (attacker_data[0][:attacker_data_size_reg],
                         attacker_data[1][:attacker_data_size_reg])

    additional_callbacks2 = [
        ShowErrorsCallback(dataset=trigger["keys"], prefix="WB Trigger")
    ]
    surr_model_reg, reg_history = whitebox_attack(
        wm_model=wm_model,
        load_model_func=load_wm_model_func,
        load_func_kwargs={"reg": reg_whitebox},
        load_func_kwargs2={
            "reg": reg_surr,
            "optimizer": RMSprop(lr=lr_surr),
            "freeze_first_layers": freeze_first_layers
        },
        trigger_set=trigger,
        train_data=attacker_data_reg,
        test_data=test_data,
        batchsize=batchsize_reg,
        epochs_reg=epochs_reg,
        additional_callbacks=additional_callbacks2,
        early_stopping_wm=early_stopping_wm_reg,  # When to stop
        patience=patience_reg,
        cache_surr_model=cache_reg_model,
        verbose=False,
        cluster=cluster)

    additional_callbacks_surr = [
        ShowErrorsCallback(dataset=trigger["keys"],
                           prefix="BB Trigger (Train)")
    ]

    # randomized blackbox
    # comment out if you do not want perform this on attacker data
    # random_selection = np.random.random_sample(attacker_data_size)
    # random_selection = (random_selection < 0.005).astype('int64')
    # random_target = np.random.randint(10, size=sum(random_selection))
    # random_index = np.where(random_selection == 1)[0]
    # attacker_data[1][random_index] = keras.utils.to_categorical(
    #     random_target, num_classes)
    # print(sum(random_selection), " attacker data is twisted...")

    surr_model, history_surr = blackbox_attack(
        surrogate_model=surr_model_reg,
        epochs_surr=epochs_surr,
        train_data=attacker_data,
        trigger_set=trigger,
        test_data=test_data,
        batchsize=batchsize_surr,
        additional_callbacks=additional_callbacks_surr,
        cache_surr_model=cache_surr_model,
        verbose=False,
        cluster=cluster)

    # After the black-box attack, try to embed the watermark again to further
    # reduce the old watermark retention.
    print("####################################################")
    print("Watermark retention BEFORE embeding new watermark...")
    print(surr_model.evaluate(trigger["keys"][0], trigger["keys"][1]))
    print(surr_model.evaluate(test_data[0], test_data[1]))
    print("####################################################")

    surr_model, history_embed, _ = wm_embed_func(
        surr_model,
        attacker_data[0],
        attacker_data[1],
        test_data[0],
        test_data[1],
        sess,
        fine_tuning=fine_tuning,
        load_wm_model_func=load_wm_model_func,
        retrain=False)

    print("####################################################")
    print("Watermark retention AFTER embeding new watermark...")
    print(surr_model.evaluate(trigger["keys"][0], trigger["keys"][1]))
    print(surr_model.evaluate(test_data[0], test_data[1]))
    print("####################################################")

    baseline_model1 = load_wm_model_func()
    baseline_model1.load_weights(weight_path)

    baseline_model2 = load_wm_model_func()
    baseline_model2.fit(attacker_data[0],
                        attacker_data[1],
                        batch_size=64,
                        epochs=5,
                        verbose=1,
                        validation_data=(test_data[0], test_data[1]))

    baseline_eval1 = baseline_model1.evaluate(trigger["keys"][0],
                                              trigger["keys"][1])[0]
    baseline_eval2 = baseline_model2.evaluate(trigger["keys"][0],
                                              trigger["keys"][1])[0]
    print("This is the baseline:", baseline_eval1)
    print("This is the baseline:", baseline_eval2)
    print(baseline_model1.evaluate(owner_data[0], owner_data[1]))

    baseline = (baseline_eval1 / 100, baseline_eval2 / 100)

    return surr_model, (history_embed, reg_history, history_surr, baseline)

