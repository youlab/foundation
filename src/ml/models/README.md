# Machine learning
This directory has code for the machine learning models used in the project

## Files
* `mlp.py`: contains multi-layer perceptron model
* `autoencoder7x.py`: contains autoencoder7x model, which is a custom variational autoencoder with both convolution and linear layers
* `mlp_network_model.py`: contains multi-layer perceptron network model, which is a custom variational autoencoder with linear layers
* `vae_bottleneck.py`: contains variational autoencoder bottleneck model, which is a custom variational autoencoder with convolution layers
* `mlp_embedding.py`: used to create embeddings of categorical information, not used in the paper
* `microbert_curve_reducer.py`: contains the model MicroBERT Curve Reducer

* the following files are related to the Neural ODE model, which was not used in the paper
    * `growth_model.py`: defines the logistic growth model
    * `neural_ode.py`: defines the neural ODE model