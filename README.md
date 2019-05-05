# gradientdescent
Simple gradient descent/ascent examples with tensorflow

1. **Optimize a function** 
* [Function of one variable](https://github.com/sgttwld/gradientdescent/blob/master/1a_GradientDescent_1d.py): Find the argument that minimizes a function of one variable. 
* [Function of two variables](https://github.com/sgttwld/gradientdescent/blob/master/1b_GradientDescent_2d.py): Find the arguments that maximize a function of two variables. 


2. **Maximize an expected value**
* [Discrete probability, exact expectation](https://github.com/sgttwld/gradientdescent/blob/master/2a_GradientDescent_prob.py): Find the discrete probability distribution that maximizes the expected value of a given function.
* [Discrete probability, approximate expectation via sampling](https://github.com/sgttwld/gradientdescent/blob/master/2b_GradientDescent_sample.py): Sampling based implementation to find the discrete probability distribution that maximizes the expected value of a given function. 
* [Continuous probability density, approximate expectation via sampling](https://github.com/sgttwld/gradientdescent/blob/master/2c_GradientDescent_cont.py): Sampling based implementation to find the parameters of a probability density that maximizes the expected value of a given utility function.

3. **Maximize Free Energy (information-theoretic bounded rationality)**
* [discrete probability, fixed prior](https://github.com/sgttwld/gradientdescent/blob/master/3a_GradientDescent_bounded.py): Find the discrete probability distribution that solves the most simple one-step free energy optimization problem of information-theoretic bounded rationality.
* [discrete probability, optimal prior (rate distortion)](https://github.com/sgttwld/gradientdescent/blob/master/3b_GradientDescent_ratedistortion.py): Find the discrete probability distribution that solves a rate distortion problem.
* [continuous density, optimal prior (rate distortion)](https://github.com/sgttwld/gradientdescent/blob/master/3c_GradientDescent_ratedistortion_cont.py): Find the parameters of a probability density that solves the rate distortion problem. 