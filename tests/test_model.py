import pytest
import torch
import torch.nn as nn
from src.model import Net

@pytest.fixture
def model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Net().to(device)

def test_parameter_count(model):
    """Test if model has less than 20K parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be < 20000"
    print("✓ Model has less than 20K parameters:", total_params)

def test_has_batchnorm(model):
    """Test if model uses batch normalization"""
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, "Model should use BatchNormalization"
    print("✓ Model uses Batch Normalization")

def test_has_dropout(model):
    """Test if model uses dropout"""
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"
    print("✓ Model uses Dropout")

def test_has_fc_layer(model):
    """Test if model uses fully connected layers"""
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())
    assert has_linear, "Model should use Fully Connected layers"
    print("✓ Model uses Fully Connected layers")

def test_forward_pass(model):
    """Test if forward pass works"""
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    input_tensor = input_tensor.to(next(model.parameters()).device)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), got {output.shape}"

def print_model_parameters(model):
    """Print parameter count for each layer"""
    print("\nParameter count breakdown:")
    total_params = 0
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name}: {param_count:,} parameters")
    print(f"Total: {total_params:,} parameters")

# Helper function for train.py to use
def run_all_tests(model):
    """Run all model architecture tests"""
    try:
        test_parameter_count(model)
        test_has_batchnorm(model)
        test_has_dropout(model)
        test_has_fc_layer(model)
        test_forward_pass(model)
        print_model_parameters(model)
        print("\nAll tests passed successfully! ✓")
        return True
    except AssertionError as e:
        print("\nTest failed:", str(e))
        return False

if __name__ == "__main__":
    # Create model instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    # Run all tests
    run_all_tests(model) 