# Cooperative Sheaf Neural Networks

Repository containing the code from the paper [Cooperative Sheaf Neural Networks](https://arxiv.org/abs/2507.00647).

![](https://github.com/ML-FGV/neural-sheaf-diffusion/blob/master/cooperative_sheaves/figures/sheaves.svg)

To reproduce our experiments, go to [this](https://github.com/andrerg02/Cooperative-Sheaf-NN/releases/tag/ICLR-Reproducibility) release and follow the instructions below. If you just wish to use our model, just install the dependencies and check the `example.ipynb` file.

## Environment setup

We used [Pyenv](https://github.com/pyenv/pyenv) to create our environment, using Python 3.10.12. After installing pyenv, run the following to set up the environment:
```bash
pyenv virtualenv 3.10.12 csnn
pyenv activate csnn
pip install -r requirements.txt
```

## Experiments

You can opt for using [Weights & Biases](https://wandb.ai/site) to track experiment metrics. After creating an account, run
```bash
wandb online && wandb login
```
or just disable it with `wandb disabled`. We provide scripts to run each dataset inside `exp/scripts`.
For instace, you can run the model on the minesweeper dataset using
```bash
sh ./exp/scripts/run_minesweeper.sh
```

## Hyperparameter details

We provide further details on sheaf-specific hyperparameters below:

```
--d                 Dimension of the sheaf stalks.
--left_weights      Whether to apply left weights in the features.
--right_weights     Whether to apply right weights in the features
--use_bias          Wheter to use an additive bias when applying the weights.
--sheaf_act         Activation function applied on the sheaf maps.
--orth              Method to learn orthogonal maps. Options are 'householder', 'matrix_exp', 'cayley', or 'euler'.
--linear_emb        Use a linear+act embedding/readout when learning the sheaf.
--gnn_type          Type of GNN to use for learning the sheaf. Options are:
                    'SAGE', 'GCN', 'GAT', 'NNConv', 'SGC', or 'SumGNN'.
--gnn_layers        Number of GNN layers to use for learning the sheaf.
--gnn_hidden        Number of hidden channels in the GNN layers.
--gnn_default       Set this to 0 to use a custom GNN setup, and 1 or 2 to reproduce experiments in the paper.
--gnn_residual      Use residual connections in the GNN layers.
--pe_size           Size of the positional encoding to use in the GNN layers.
--conformal         Whether to learn conformal restriction maps. If set to
                    False, the model learns in and out flat bundles
```

## Further details

To facilitate integration with other experimental setups, the file `model.py` contains an implementation of the model using the `MessagePassing` class, which can be instantiated in a manner similar to other PyG models such as GCN and GAT. This file is self-contained and does not depend on any other modules in the reproducibility release.

