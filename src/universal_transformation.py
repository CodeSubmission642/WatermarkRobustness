import keras
import numpy as np
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K

from numpy import linalg as LA

from scipy.optimize import minimize, rosen_der, rosen

from matplotlib import pyplot as plt
from src.mnist_main import get_mnist_model_and_data


def compute_universal_transformation(X,
                                     model,
                                     sess,
                                     ref_model=None,
                                     epochs=2,
                                     eps=0.25):
    """ Finds the minimal universal transformation for some inputs, i.e. the subspace S for X_v
    """
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap)
    fgsm_params = {'eps': eps}

    adv_x_op = fgsm.generate(model.inputs[0], **fgsm_params)
    preds_adv_op = model(adv_x_op)
    preds_op = model(model.inputs[0])

    x0 = None
    for i in range(epochs):
        adv_x, preds_adv_one_hot, preds_one_hot = sess.run(
            [adv_x_op, preds_adv_op, preds_op],
            feed_dict={model.inputs[0]: X})

        # Get only true adversarial examples (misclassification compared to original sample)
        preds_adv = np.argmax(preds_adv_one_hot, axis=1)
        preds = np.argmax(preds_one_hot, axis=1)
        pred_comp = (preds_adv == preds).astype('int32')
        true_adv = np.where(pred_comp == 0)[0]

        # Obtain the perturbation vectors
        v_per = adv_x[true_adv] - X[true_adv]

        # Find minimal universal perturbation v' s.t. arg_min_{v'}(|v-v'|) (and |v'|<eps)
        def f(x):
            return LA.norm(np.mean(np.abs(v_per - x), axis=0))

        if x0 is None:
            x0 = np.reshape(v_per[0], (28, 28))
        cons = ({'type': 'ineq', 'fun': lambda x: 200-LA.norm(x)})

        res = minimize(f, x0.flatten(), method='SLSQP', jac=rosen_der, constraints=cons, options={'maxiter': 10, 'disp': True})
        x0 += np.reshape(res.x, (28, 28))

        # Evaluate accuracy of perturbation
        pred_per = np.argmax(model.predict(X[true_adv]+np.reshape(x0, (28, 28, 1))), axis=1)
        per_acc = np.mean((pred_per != preds[true_adv]).astype('int32'))
        print("Iter {}: Perturbation Accuracy: {}".format(i, per_acc))

        if ref_model is not None:
            pred_ref = np.argmax(model2.predict(X[true_adv]), axis=1)
            pred_per_ref = np.argmax(model2.predict(X[true_adv]+np.reshape(x0, (28, 28, 1))), axis=1)
            per_acc2 = np.mean((pred_ref != pred_per_ref).astype('int32'))
            print("Iter {}, Ref Model: Perturbation Accuracy: {}".format(i, per_acc2))

def run_universal_transformation(sess):
    print(">> Universal Transformation")
    (x_train, y_train), (x_test, y_test), model = get_mnist_model_and_data(epochs=3, split=(0, 20000))
    train2, test2, model2 = get_mnist_model_and_data(epochs=3, split=(20000, 40000))
    compute_universal_transformation(x_train[:10], model, sess, ref_model=model2)

if __name__ == "__main__":
    sess = tf.Session()
    K.set_session(sess)

    # run_universal_transformation(sess)

    (x_train, y_train), (x_test, y_test), model = get_mnist_model_and_data(epochs=3, split=(0, 20000))
    train2, test2, model2 = get_mnist_model_and_data(epochs=3, split=(20000, 40000))

    preds1 = np.argmax(model.predict(x_test), axis=1)
    preds2 = np.argmax(model2.predict(x_test), axis=1)

    ground_truth = (np.argmax(y_test, axis=1) == preds1).astype('int32')
    similarity = (preds1 != preds2).astype('int32')
    acc = np.mean(ground_truth * similarity)
    print(acc)

