## Adversarial Watermark Embedding and Attacks on Them

### All the code is placed in **src/adversarial_main.py**

### 1. Frontier Stitching Zerobit (https://arxiv.org/pdf/1711.01894.pdf)

#### Embeding

The watermark embedding is implemented by the function **zerobit_embed**. It takes in a model, training data, test data and also a tf.session. The outputs are the embedded model, history lists for visualization purpose and a trigger set as the key of the watermark embedding.

Some important parameters can be changed:
- **eps**, by default is 0.25 it represents the strength of adversarial generator distorts the input to generate adversarial examples, higher the number gives higher strength.
- **key_length** is 100 by default, it is the size of the trigger set.
- **wm_epoch** 12 by default. The number epochs for fine-tuning the model on the adversarials and also the number of epoches to train the model on the original training dataset.
- **fine_tuning** (not used) True by default. The code for fine_tuning is commented out now, but it is functional and can be uncommented to use. The code shuffles the trigger set into the original training data and train these combined dataset together instead of training them separately.

#### Extraction

Extraction is done by **zerobit_extract**. It does a hypothesis testing with null hypothesis of the model is not watermarked and trying to reject the null hypothesis at a confidence level of 5% based on the evaluation result of the model on the trigger set.

### 2. Blackmarks Multibits (https://arxiv.org/pdf/1904.00344.pdf)

#### Embedding

The embedding function is enclosed in **blackmarks_embed** function and can be obtained by passing a signature to the enclosing function. I did this so that the watermark embedding functions for both this and frontier stitching can follow the same API. Please be aware that the implementation of Blackmarks is based on my understanding of the paper. The authors are a little hand-waving on some implementation details.

The parameters are identitcal to the ones for frontier stitching.

#### Extraction

It evaluates the model on trigger set and returns positive only if the model gets 100% accuracy on the trigger set.

**(Reference whitebox and blackbox attack functions for an example usage of these embedding functions)**

### 3. Adversarial Whitebox and Blackbox Attack

The two attack functions are very similar to the whitebox and blackbox attack in other notebooks with exceptions:

- **rand_bb:** by default is False, it can toggle whether you want to do a randomization on blackbox attacker data after each epoch. randomization means that after each epoch, makes 0.5% of labels of the training data incorrect. There's also another version that only does the randomization once before the blackbox attack.
- At the end of the attack both attacks also try to embed a new watermark onto the model as an attempt to reduce watermark retention  but the result is not very promising.

**(Reference the demo notebook for an example of using these attack functions)**