import pytest
import numpy as np

def test_est_curve_quantiles(self):
    # Arrange
    correct_response = (40, 50, 60)
    xv = np.arange(0,1000)
    test_data = {
        'x': xv,
        'y': (2.*np.pi*10**2)**-0.5 * np.exp(-1.*(xv - 50)/(2.*10**2))
    }
    # Act
    response = 

