
import numpy as np
from bokbokbok.utils import clip_sigmoid

def test_clip_sigmoid():
    assert np.allclose(a=clip_sigmoid(np.array([100, 0, -100])),
                       b=[1 - 1e-15, 0.5, 1e-15])