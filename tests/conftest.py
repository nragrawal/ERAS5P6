import pytest
import torch

@pytest.fixture(scope="session")
def device():
    """Fixture to handle CPU/GPU device consistently"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 