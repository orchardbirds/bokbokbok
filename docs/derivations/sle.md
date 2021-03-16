## Squared Log Error

The equation for Squared Log Error is:

<img src="https://latex.codecogs.com/svg.latex?L_{SLE}&space;=&space;\frac{1}{2}(\log(\hat{y}&space;&plus;&space;1)&space;-&space;\log(y&space;&plus;&space;1))^{2}" title="L_{SLE} = \frac{1}{2}(\log(\hat{y} + 1) - \log(y + 1))^{2}" />

We calculate the Gradient:

<img src="https://latex.codecogs.com/svg.latex?G_{SLE}&space;=&space;\frac{\log(\hat{y}&space;&plus;&space;1)&space;-&space;\log(y&space;&plus;&space;1)}{\hat{y}&space;&plus;&space;1}" title="G_{SLE} = \frac{\log(\hat{y} + 1) - \log(y + 1)}{\hat{y} + 1}" />

We also need to calculate the Hessian:

<img src="https://latex.codecogs.com/svg.latex?H_{SLE}&space;=&space;\frac{-&space;\log(\hat{y}&space;&plus;&space;1)&space;&plus;&space;\log(y&space;&plus;&space;1)&space;&plus;&space;1}{(\hat{y}&space;&plus;&space;1)^{2}}" title="H_{SLE} = \frac{- \log(\hat{y} + 1) + \log(y + 1) + 1}{(\hat{y} + 1)^{2}}" />