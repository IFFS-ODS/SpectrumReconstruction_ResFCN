# SpectrumReconstruction_ResFCN

## Features

- **Residual Architecture**: Implements both identity and dense residual blocks for better performance
- **Regularization**: Includes L2 regularization to prevent overfitting
- **Custom Loss Function**: Combines MSE with L1 loss for improved prediction accuracy
- **Learning Rate Scheduling**: Dynamic learning rate adjustment during training
- **GPU Support**: Optimized for GPU acceleration
- **Early Stopping**: Prevents overfitting by monitoring validation loss

## Requirements

- CUDA 11.2
- Python 3.8.x
- TensorFlow 2.9.0 (GPU version)
- Keras 2.9.0
- NumPy 1.24.3
- Matplotlib 3.7.2
- Scikit-learn 1.3.0
- Pandas 2.0.3

## Installation

```bash
git clone https://github.com/yourusername/ResFCNet.git
cd ResFCNet
pip install -r requirements.txt
```

## Usage

The model is designed to work with numerical feature inputs and outputs. Basic usage:

```python
from ResFCNet import ImprovedResNet

# Create the model
model = ImprovedResNet()

# Compile with custom loss function
model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, 
                    validation_split=0.1)
```

## Model Architecture

The network consists of:
- Multiple residual blocks with increasing units (256, 512, 1024)
- Linear output layers for regression tasks
- Custom loss function combining MSE and L1 loss
- Adam optimizer with scheduled learning rate

## Performance Visualization

The code automatically generates:
- Training and validation loss curves
- Comparison plots between predicted and actual values for test samples with lowest validation loss

## License

[MIT License](LICENSE)

## Contact

For questions or feedback, please open an issue or contact us by email.
