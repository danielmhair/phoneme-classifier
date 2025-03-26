import pytest
from src.main import my_function

def test_my_function():
    assert my_function(2, 3) == 5
    assert my_function(0, 0) == 0
    assert my_function(-1, 1) == 0
    # Add more test cases here