## A Note About Gradients in Classification Problems

For the gradient boosting packages we [have to calculate the gradient of the Loss function with respect to the marginal probabilites](https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py).

In this case, we must calculate

<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;z}&space;=&space;\frac{\partial&space;L}{\partial&space;\hat{y}}&space;\cdot&space;\frac{\partial&space;\hat{y}}{\partial&space;z}" title="\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}" />



The Hessian is similarly calculated:

<img src="https://latex.codecogs.com/svg.latex?\frac{\partial^{2}&space;L}{\partial&space;z^{2}}&space;=&space;\frac{\partial}{\partial&space;z}[\frac{\partial&space;L}{\partial&space;\hat{y}}&space;\cdot&space;\frac{\partial&space;\hat{y}}{\partial&space;z}]" title="\frac{\partial^{2} L}{\partial z^{2}} = \frac{\partial}{\partial z}[\frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}]" />


**Where y-hat is the sigmoid function, unless stated otherwise**:

<img src="https://latex.codecogs.com/svg.latex?\hat{y}&space;=&space;\sigma(z)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}" />

We will make use of the following property for the calculations of the Gradients and Hessians:

<img src="https://latex.codecogs.com/svg.latex?\frac{\partial&space;\hat{y}}{\partial&space;z}&space;=&space;\hat{y}&space;\cdot&space;(1&space;-&space;\hat{y})" title="\frac{\partial \hat{y}}{\partial z} = \hat{y} \cdot (1 - \hat{y})" />