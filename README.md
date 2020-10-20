# Content

This reposity contains supplementary source code for the paper "On the Robustness of the 
Backdoor-based Watermarking in Deep Neural Networks". 
We distribute our code-base to replicate the black-box and white-box watermark removal 
attacks presented in the paper on MNIST and CIFAR-10. 


# Setup
We use Tensorflow 1.15 and Keras 2.3.0. 
It is advisable to install Tensorflow with GPU support, as the experiments may take 
several hours.
More detailed descriptions for the experiments can be found in the notebooks. 

1.) Clone the repository 

```
 git clone https://github.com/CodeSubmission642/WatermarkRobustness
```
2.) Install the requirements
```
 cd WatermarkRobustness
 pip install -r requirements.txt
```

3.) Start the Jupyter notebook and connect to the notebook in your browser.
```
 jupyter notebook 
```

