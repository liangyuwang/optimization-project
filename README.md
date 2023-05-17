# Optimization Algorithm Comparison

This project compares different optimization algorithms for training a multilayer perceptron (MLP) with one hidden layer. The implemented algorithms include LBFGS, BFGS, Gradient Descent, and Steepest Descent. The project also provides an implementation of the MLP model, cross-entropy loss function, and a method to load the MNIST dataset. We implemented all the code manually, except for the auto-grad part, which leverages the PyTorch library for efficiency and compatibility with GPU acceleration.

## Files

The project consists of the following files:

- `optim.py`: Contains the implementation of the optimization algorithms, including LBFGS, BFGS, Gradient Descent, and Steepest Descent.
- `models.py`: Implements the MLP model with one hidden layer.
- `losses.py`: Provides the implementation of the cross-entropy loss function.
- `dataset.py`: Contains code for loading the MNIST dataset.
- `main.py`: The main file to run the code. Accepts command-line arguments to specify the model, optimizer, batch size, number of epochs, hidden size, and device.
- `images/`: Directory containing experimental result images.

## Running the Code

To run the code, use the following command:

```shell
python main.py --model=mlp2 --optimizer=bfgs --batch_size=10000 --num_epoch=100 --hidden_size=200 --device=cuda:0
```

You can modify the command-line arguments to experiment with different settings. Here's a breakdown of the available arguments:

- `--model`: Specifies the model architecture. Options include `mlp1` and `mlp2`.
- `--optimizer`: Specifies the optimization algorithm. Options include `bfgs`, `lbfgs`, `gd`, and `sd`.
- `--batch_size`: Sets the batch size for training.
- `--num_epoch`: Specifies the number of epochs for training.
- `--hidden_size`: Sets the size of the hidden layer in the MLP.
- `--device`: Specifies the device to run the code on, such as `cpu` or `cuda:0`.

## Results

The experimental results are saved in the `images/` directory. You can refer to these images to analyze and compare the performance of different optimization algorithms.

## Dependencies

The code relies on the PyTorch library for automatic differentiation and GPU support. Ensure that you have PyTorch installed before running the code.

## Acknowledgments

This project implements optimization algorithms based on well-known techniques and builds upon the PyTorch library for training neural networks. We acknowledge the contributions of the PyTorch community and the developers of the optimization algorithms.

