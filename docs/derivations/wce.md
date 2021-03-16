## Weighted Cross Entropy Loss

Weighted Cross Entropy applies a scaling parameter *alpha*  to [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression),
allowing us to penalise false positives or false negatives more harshly. If you want false
positives to be penalised more than false negatives, *alpha* must be greater than 1. Otherwise,
it must be less than 1. 

The equations for Binary and Weighted Cross Entropy Loss are the following:

<img src="https://latex.codecogs.com/svg.latex?L_{BCE}&space;=&space;-y&space;\log(\hat{y}(z))&space;-&space;(1&space;-&space;y)\log(1&space;-&space;\hat{y}(z))" title="L_{BCE} = -y \log(\hat{y}(z)) - (1 - y)\log(1 - \hat{y}(z))" />

<img src="https://latex.codecogs.com/svg.latex?L_{WCE}&space;=&space;-&space;\alpha&space;y&space;\log(\hat{y}(z))&space;-&space;(1&space;-&space;y)\log(1&space;-&space;\hat{y}(z))" title="L_{WCE} = - \alpha y \log(\hat{y}(z)) - (1 - y)\log(1 - \hat{y}(z))" />

We calculate the Gradient:

<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;G_{WCE}(z)&space;&&space;=&space;\frac{\partial&space;L_{WCE}}{\partial&space;z}&space;=&space;\frac{\partial&space;L}{\partial&space;\hat{y}}&space;\cdot&space;\frac{\partial&space;\hat{y}}{\partial&space;z}&space;\\&space;&&space;=&space;\frac{\partial&space;L}{\partial&space;\hat{y}}&space;\cdot&space;\hat{y}&space;\cdot&space;(1&space;-&space;\hat{y})&space;\\&space;&&space;=&space;[\frac{-\alpha&space;y}{\hat{y}}&space;&plus;&space;\frac{1&space;-&space;y}{1&space;-&space;\hat{y}}]\cdot&space;\hat{y}&space;\cdot&space;(1&space;-&space;\hat{y})&space;\\&space;&&space;=&space;y\hat{y}(\alpha&space;-&space;1)&space;&plus;&space;\hat{y}&space;-&space;\alpha&space;y&space;\end{aligned}" title="\begin{aligned} G_{WCE}(z) & = \frac{\partial L_{WCE}}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \\ & = \frac{\partial L}{\partial \hat{y}} \cdot \hat{y} \cdot (1 - \hat{y}) \\ & = [\frac{-\alpha y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}]\cdot \hat{y} \cdot (1 - \hat{y}) \\ & = y\hat{y}(\alpha - 1) + \hat{y} - \alpha y \end{aligned}" />


We also need to calculate the Hessian:

<img src="https://latex.codecogs.com/svg.latex?\begin{aligned}&space;H_{WCE}(z)&space;&&space;=&space;\frac{\partial&space;G_{WCE}(z)}{\partial&space;z}&space;=&space;\frac{\partial&space;G_{WCE}}{\partial&space;\hat{y}}&space;\cdot&space;\frac{\partial&space;\hat{y}}{\partial&space;z}&space;\\&space;&&space;=&space;\frac{\partial&space;G_{WCE}(z)}{\partial&space;\hat{y}}&space;\cdot&space;\hat{y}&space;\cdot&space;(1&space;-&space;\hat{y})&space;\\&space;&&space;=&space;(y(\alpha&space;-&space;1)&space;&plus;&space;1)&space;\cdot&space;\hat{y}(1&space;-&space;\hat{y})&space;\end{aligned}" title="\begin{aligned} H_{WCE}(z) & = \frac{\partial G_{WCE}(z)}{\partial z} = \frac{\partial G_{WCE}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \\ & = \frac{\partial G_{WCE}(z)}{\partial \hat{y}} \cdot \hat{y} \cdot (1 - \hat{y}) \\ & = (y(\alpha - 1) + 1) \cdot \hat{y}(1 - \hat{y}) \end{aligned}" />


By setting *alpha* = 1 we obtain the Gradient and Hessian for Binary Cross Entropy Loss, as expected.