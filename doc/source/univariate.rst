Univariate Distributions
==========================

*Univariate distributions* are the distributions whose samples are scalars (*e.g.* integers or real values). *Distributions* provides a large collection of univariate distributions, as listed below.

Discrete Distributions
------------------------

All discrete univariate distribution types are subtypes of *DiscreteUnivariateDistribution*. Each sample from a discrete univariate distribution is an integer (of type ``Int``).


Bernoulli Distribution 
~~~~~~~~~~~~~~~~~~~~~~~

A *Bernoulli distribution* is parameterized by a success rate p, which takes value 1 with probability p and 0 with probability 1-p. 

.. code-block:: julia

    Bernoulli()    # Bernoulli distribution with p = 0.5
    Bernoulli(p)   # Bernoulli distribution with success rate p

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~

A *Binomial distribution* characterizes the number of successes in a sequence of independent trials. It has two parameters: n, the number of trials, and p, the success rate. 

.. code-block:: julia

    Binomial()      # Binomial distribution with n = 1 and p = 0.5
    Binomial(n)     # Binomial distribution for n trials with success rate p = 0.5
    Binomial(n, p)  # Binomial distribution for n trials with success rate p

Categorical Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

A *Categorical distribution* is parameterized by a probability vector p. Particularly, ``p[k]`` is the probability of drawing ``k``. 

.. code-block:: julia

    Categorical(p)   # Categorical distribution with probability vector p

Here, ``p`` must be a real vector, of which all components are nonnegative and sum to one. 

**Note:** The input vector ``p`` is directly used as a field of the constructed distribution, without being copied. 


Discrete Uniform Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *Discrete uniform distribution* is a uniform distribution over a consecutive sequence of integers. 

.. code-block:: julia

    DiscreteUniform(a, b)   # a uniform distribution over {a, a+1, ..., b}


Geometric Distribution
~~~~~~~~~~~~~~~~~~~~~~~

A *Geometric distribution* characterizes the number of failures before the first success in a sequence of independent Bernoulli trials. 

.. code-block:: julia

    Geometric()    # Geometric distribution with success rate 0.5
    Geometric(p)   # Geometric distribution with success rate p


Hypergeometric Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A *Hypergeometric distribution* describes the number of successes in *n* draws without replacement from a finite population containing *s* successes and *f* failures.

.. code-block:: julia

    Hypergeometric(s, f, n)  # Hypergeometric distribution for a population with 
                             # s successes and f failures, and a sequence of n trials.


Negative Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                             

A *Negative binomial distribution* describes the number of failures before the r-th success in a sequence of independent trials. It is parameterized by r, the number of successes, and p, the success rate. 

.. code-block:: julia
    
    NegativeBinomial()        # Negative binomial distribution with r = 1 and p = 0.5
    NegativeBinomial(r, p)    # Negative binomial distribution with r successes and success rate p


Poisson Distribution
~~~~~~~~~~~~~~~~~~~~~

A *Poisson distribution* descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence.

.. code-block:: julia

    Poisson()            # Poisson distribution with rate parameter 1
    Poisson(lambda)      # Poisson distribution with rate parameter lambda


Skellam Distribution
~~~~~~~~~~~~~~~~~~~~~

A *Skellam distribution* describes the difference between two independent Poisson variables.

.. code-block:: julia

    Skellam(mu1, mu2)   # Skellam distribution for the difference between two Poisson variables,
                        # respectively with expected values mu1 and mu2.



Continuous Distributions
-------------------------






