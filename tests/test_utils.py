
from bokbokbok.utils.functions import say_hello
import numpy as np
from bokbokbok.utils import clip_sigmoid

# def test_clip_sigmoid():
#     assert clip_sigmoid(np.array(np.float(0.2342))) == 0

def test_say_hello():
    assert say_hello() == "hi"