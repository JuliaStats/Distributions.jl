.. _univariates:

Univariate Distributions
==========================

*Univariate distributions* are the distributions whose variate forms are ``Univariate`` (*i.e* each sample is a scalar). Abstract types for univariate distributions:

.. code-block:: julia

    typealias UnivariateDistribution{S<:ValueSupport} Distribution{Univariate,S}

    typealias DiscreteUnivariateDistribution   Distribution{Univariate, Discrete}
    typealias ContinuousUnivariateDistribution Distribution{Univariate, Continuous}


Common Interface
------------------

A series of methods are implemented for each univariate distribution, which provide useful functionalities such as moment computation, pdf evaluation, and sampling (*i.e.* random number generation).

Computation of statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let ``d`` be a distribution:

.. function:: params(d)

    Return a tuple of parameters. 

    **Note:** Let ``d`` be a distribution of type ``D``, then ``D(params(d)...)`` will construct exactly the same distribution as ``d``.

.. function:: mean(d)

    Return the expectation of distribution ``d``.

.. function:: var(d)

    Return the variance of distribution ``d``.

.. function:: std(d)

    Return the standard deviation of distribution ``d``, i.e. ``sqrt(var(d))``.

.. function:: median(d)

    Return the median value of distribution ``d``.

.. function:: modes(d)    

    Return an array of all modes of ``d``. 

.. function:: mode(d)

    Return the mode of distribution ``d``. If ``d`` has multiple modes, it returns the first one, i.e. ``modes(d)[1]``.

.. function:: skewness(d)

    Return the skewness of distribution ``d``.

.. function:: kurtosis(d)

    Return the excess kurtosis of distribution ``d``.

.. function:: isplatykurtic(d)

    Return whether ``d`` is platykurtic (*i.e* ``kurtosis(d) > 0``).

.. function:: isleptokurtic(d)

    Return whether ``d`` is leptokurtic (*i.e* ``kurtosis(d) < 0``).

.. function:: ismesokurtic(d)

    Return whether ``d`` is leptokurtic (*i.e* ``kurtosis(d) == 0``).

.. function:: entropy(d)

    Return the entropy value of distribution ``d``.

.. function:: entropy(d, base)

    Return the entropy value of distribution ``d``, w.r.t. a given base. 

.. function:: mgf(d, t)

    Evaluate the moment generating function of distribution ``d``.

.. function:: cf(d, t)

    Evaluate the characteristic function of distribution ``d``. 

Probability Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

.. function:: insupport(d, x)

    When ``x`` is a scalar, it returns whether x is within the support of ``d``. 
    When ``x`` is an array, it returns whether every element in x is within the support of ``d``. 

.. function:: pdf(d, x)

    The pdf value(s) evaluated at ``x``.

.. function:: pdf(d, rgn)

    Get/compute the probabilities over a range of values. Here, ``rgn`` should be in the form of ``a:b``.

    **Note:** computing the probabilities over a contiguous range of values can take advantage of the recursive relations between probability masses and thus is often more efficient than computing these probabilities individually.

.. function:: pdf(d)

    Get/compute the entire probability vector of ``d``. This is equivalent to ``pdf(d, minimum(d):maximum(d))``.

    **Note:** this method is only defined for *bounded* distributions.


.. function:: logpdf(d, x)

    The logarithm of the pdf value(s) evaluated at x, i.e. ``log(pdf(x))``. 

    **Note:** The internal implementation may directly evaluate logpdf instead of first computing pdf and then taking the logarithm, for better numerical stability or efficiency.

.. function:: loglikelihood(d, x)

    The log-likelihood of distribution ``d`` w.r.t. all samples contained in array ``x``.

.. function:: cdf(d, x)

    The cumulative distribution function evaluated at ``x``.

.. function:: logcdf(d, x)        

    The logarithm of the cumulative function value(s) evaluated at ``x``, i.e. ``log(cdf(x))``.

.. function:: ccdf(d, x)

    The complementary cumulative function evaluated at ``x``, i.e. ``1 - cdf(d, x)``.

.. function:: logccdf(d, x)

    The logarithm of the complementary cumulative function values evaluated at x, i.e. ``log(ccdf(x))``.

.. function:: quantile(d, q)

    The quantile value. Let ``x = quantile(d, q)``, then ``cdf(d, x) = q``.

.. function:: cquantile(d, q)

    The complementary quantile value, i.e. ``quantile(d, 1-q)``.

.. function:: invlogcdf(d, lp)

    The inverse function of logcdf. 

.. function:: invlogccdf(d, lp)

    The inverse function of logccdf.    


Vectorized evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Vectorized computation and inplace vectorized computation are supported for the following functions:

* ``pdf``
* ``logpdf``
* ``cdf``
* ``logcdf``
* ``ccdf``
* ``logccdf``
* ``quantile``
* ``cquantile``
* ``invlogcdf``
* ``invlogccdf``

For example, when ``x`` is an array, then ``r = pdf(d, x)`` returns an array ``r`` of the same size, such that ``r[i] = pdf(d, x[i])``. One can also use ``pdf!`` to write results to pre-allocated storage, as ``pdf!(r, d, x)``. 


Sampling (Random number generation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: rand(d)

    Draw a sample from d

.. function:: rand(d, n)

    Return a vector comprised of n independent samples from the distribution ``d``.

.. function:: rand(d, dims)

    Return an array of size dims that is filled with independent samples from the distribution ``d``.            

.. function:: rand!(d, arr)

    Fills a pre-allocated array ``arr`` with independent samples from the distribution ``d``.


List of Distributions
----------------------

*Distributions* provides a large collection of univariate distributions. Here is a brief list:

* Discrete univariate distributions:

    - :ref:`bernoulli`
    - :ref:`binomial`
    - :ref:`categorical`
    - :ref:`discreteuniform`
    - :ref:`geometric`
    - :ref:`hypergeometric`
    - :ref:`negativebinomial`
    - :ref:`poisson`
    - :ref:`skellam`

* Continuous univariate distributions:

    - :ref:`arcsine`
    - :ref:`beta`
    - :ref:`betaprime`
    - :ref:`cauchy`
    - :ref:`chi`
    - :ref:`chisquare`
    - :ref:`erlang`
    - :ref:`exponential`
    - :ref:`fdist`
    - :ref:`frechet`
    - :ref:`gamma`
    - :ref:`gumbel`
    - :ref:`inversegamma`
    - :ref:`inversegaussian`
    - :ref:`laplace`
    - :ref:`levy`
    - :ref:`logistic`
    - :ref:`lognormal`
    - :ref:`normal`
    - :ref:`pareto`
    - :ref:`rayleigh`
    - :ref:`tdist`
    - :ref:`triangular`
    - :ref:`uniform`
    - :ref:`vonmises`
    - :ref:`weibull`


Discrete Distributions
------------------------

All discrete univariate distribution types are subtypes of *DiscreteUnivariateDistribution*. Each sample from a discrete univariate distribution is an integer (of type ``Int``).

.. _bernoulli:

Bernoulli Distribution 
~~~~~~~~~~~~~~~~~~~~~~~

A `Bernoulli distribution <http://en.wikipedia.org/wiki/Bernoulli_distribution>`_ is parameterized by a success rate :math:`p`, which takes value 1 with probability :math:`p` and 0 with probability :math:`1-p`. 

.. math:: 

    P(X = k) = \begin{cases}
        p & (k = 0) \\
        1 - p & (k = 1)
    \end{cases} 

.. code-block:: julia

    Bernoulli()    # Bernoulli distribution with p = 0.5
    Bernoulli(p)   # Bernoulli distribution with success rate p

    params(d)      # Get the parameters, i.e. (p,)
    succprob(d)    # Get the success rate, i.e. p
    failprob(d)    # Get the failure rate, i.e. 1 - p


.. _binomial:

Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~

A `Binomial distribution <http://en.wikipedia.org/wiki/Binomial_distribution>`_ characterizes the number of successes in a sequence of independent trials. It has two parameters: :math:`n`, the number of trials, and :math:`p`, the success rate. 

.. math::

    P(X = k) = {n \choose k}p^k(1-p)^{n-k},  \quad \text{ for } 0 \le k \le n

.. code-block:: julia

    Binomial()      # Binomial distribution with n = 1 and p = 0.5
    Binomial(n)     # Binomial distribution for n trials with success rate p = 0.5
    Binomial(n, p)  # Binomial distribution for n trials with success rate p

    params(d)       # Get the parameters, i.e. (n, p)
    ntrials(d)      # Get the number of trials, i.e. n
    succprob(d)     # Get the success rate, i.e. p
    failprob(d)     # Get the failure rate, i.e. 1 - p


.. _categorical:

Categorical Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

A `Categorical distribution <http://en.wikipedia.org/wiki/Categorical_distribution>`_ is parameterized by a probability vector :math:`p` (of length ``K``). 

.. math::

    P(X = k) = \begin{cases}
        p[k] & (1 \le k \le K) \\
        0 & (\text{otherwise})
    \end{cases}

.. code-block:: julia

    Categorical(p)   # Categorical distribution with probability vector p

    params(d)        # Get the parameters, i.e. (p,)
    probs(d)         # Get the probability vector, i.e. p
    ncategories(d)   # Get the number of categories, i.e. K

Here, ``p`` must be a real vector, of which all components are nonnegative and sum to one. 

**Note:** The input vector ``p`` is directly used as a field of the constructed distribution, without being copied. 


.. _discreteuniform:

Discrete Uniform Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A `Discrete uniform distribution <http://en.wikipedia.org/wiki/Uniform_distribution_(discrete)>`_ is a uniform distribution over a consecutive sequence of integers between :math:`a` and :math:`b`. 

.. math::

    P(X = k) = \begin{cases}
        1 / (b - a + 1) & (a \le k \le b) \\
        0 & (\text{otherwise}) 
    \end{cases}

.. code-block:: julia

    DiscreteUniform(a, b)   # a uniform distribution over {a, a+1, ..., b}

    params(d)       # Get the parameters, i.e. (a, b)
    span(d)         # Get the span of the support, i.e. (b - a + 1)
    probval(d)      # Get the probability value, i.e. 1 / (b - a + 1)
    minimum(d)      # Return a
    maximum(d)      # Return b


.. _geometric:

Geometric Distribution
~~~~~~~~~~~~~~~~~~~~~~~

A `Geometric distribution <http://en.wikipedia.org/wiki/Geometric_distribution>`_ characterizes the number of failures before the first success in a sequence of independent Bernoulli trials with success rate :math:`p`. 

.. math::

    P(X = k) = p (1 - p)^k

.. code-block:: julia

    Geometric()    # Geometric distribution with success rate 0.5
    Geometric(p)   # Geometric distribution with success rate p

    params(d)      # Get the parameters, i.e. (p,)
    succprob(d)    # Get the success rate, i.e. p
    failprob(d)    # Get the failure rate, i.e. 1 - p


.. _hypergeometric:

Hypergeometric Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A `Hypergeometric distribution <http://en.wikipedia.org/wiki/Hypergeometric_distribution>`_ describes the number of successes in :math:`n` draws without replacement from a finite population containing :math:`s` successes and :math:`f` failures. 

.. math::

    P(X = k) = {{{s \choose k} {f \choose {n-k}}}\over {s+f \choose n}}, \quad k \in [\max(0, n - f), \min(n, s)]

.. code-block:: julia

    Hypergeometric(s, f, n)  # Hypergeometric distribution for a population with 
                             # s successes and f failures, and a sequence of n trials.

    params(d)       # Get the parameters, i.e. (s, f, n)


.. _negativebinomial:

Negative Binomial Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A `Negative binomial distribution <http://en.wikipedia.org/wiki/Negative_binomial_distribution>`_ describes the number of failures before the :math:`r`-th success in a sequence of independent trials. It is parameterized by :math:`r`, the number of successes, and :math:`p`, the success rate. 

.. math::

    P(X = k) = {k + r - 1 \choose k} p^r (1 - p)^k, \quad \text{ for } k \ge 0

.. code-block:: julia
    
    NegativeBinomial()        # Negative binomial distribution with r = 1 and p = 0.5
    NegativeBinomial(r, p)    # Negative binomial distribution with r successes and success rate p

    params(d)       # Get the parameters, i.e. (r, p)
    succprob(d)     # Get the success rate, i.e. p
    failprob(d)     # Get the failure rate, i.e. 1 - p


.. _poisson:

Poisson Distribution
~~~~~~~~~~~~~~~~~~~~~

A `Poisson distribution <http://en.wikipedia.org/wiki/Poisson_distribution>`_ descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence :math:`\lambda`.

.. math::

    P(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}, \quad \text{ for } k \ge 0

.. code-block:: julia

    Poisson()            # Poisson distribution with rate parameter 1
    Poisson(lambda)      # Poisson distribution with rate parameter lambda

    params(d)        # Get the parameters, i.e. (lambda,)
    mean(d)          # Get the mean arrival rate, i.e. lambda


.. _skellam:

Skellam Distribution
~~~~~~~~~~~~~~~~~~~~~

A `Skellam distribution <http://en.wikipedia.org/wiki/Skellam_distribution>`_ describes the difference between two independent Poisson variables, respectively with rate :math:`\mu_1` and :math:`\mu_2`.

.. math::

    P(X = k) = e^{-(\mu_1 + \mu_2)} \left( \frac{\mu_1}{\mu_2} \right)^{k/2} I_k(2 \sqrt{\mu_1 \mu_2})

Here, :math:`I_k` is the modified Bessel function of the first kind.  

.. code-block:: julia

    Skellam(mu1, mu2)   # Skellam distribution for the difference between two Poisson variables,
                        # respectively with expected values mu1 and mu2.

    params(d)           # Get the parameters, i.e. (mu1, mu2)



Continuous Distributions
-------------------------

All discrete univariate distribution types are subtypes of *ContinuousUnivariateDistribution*. Each sample from a discrete univariate distribution is a real-valued scalar (of type ``Float64``).

.. _arcsine:

Arcsine Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability density function of an `Arcsine distribution <http://en.wikipedia.org/wiki/Arcsine_distribution>`_ is:

.. math::

    f(x) = \frac{1}{\pi \sqrt{x (1 - x)}}, \quad x \in [0, 1]

.. code-block:: julia
    
    Arcsine()    # Arcsine distribution

.. _beta:

Beta Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Beta distribution <http://en.wikipedia.org/wiki/Beta_distribution>`_ with shape parameters α and β is:

.. math::

    f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} 
    x^{\alpha - 1} (1 - x)^{\beta - 1}, \quad x \in [0, 1]

.. code-block:: julia
    
    Beta(a, b)    # Beta distribution with shape parameters a and b


.. _betaprime:

Beta Prime Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Beta prime distribution <http://en.wikipedia.org/wiki/Beta_prime_distribution>`_ with shape parameters α and β is:

.. math::

    f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} 
    x^{\alpha - 1} (1 + x)^{- (\alpha + \beta)}, \quad x > 0

.. code-block:: julia
    
    BetaPrime(a, b)    # Beta prime distribution with shape parameters a and b

.. _cauchy:

Cauchy Distribution
~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Cauchy distribution <http://en.wikipedia.org/wiki/Cauchy_distribution>`_ with location μ and scale θ is:

.. math::

    f(x; \mu, \theta) = \frac{1}{\pi \theta \left(1 + \left(\frac{x - \mu}{\theta} \right)^2 \right)}

.. code-block:: julia

    Cauchy()         # Standard Cauchy distribution (location = 0.0, scale = 1.0)
    Cauchy(u)        # Cauchy distribution with location u and unit scale
    Cauchy(u, s)     # Cauchy distribution with location u and scale s

.. _chi:

Chi Distribution
~~~~~~~~~~~~~~~~~

The `Chi distribution <http://en.wikipedia.org/wiki/Chi_distribution>`_ with k degrees of freedom is the distribution of the square root of the sum of squares of k independent variables that are normally distributed. The probability density function is:

.. math::

    f(x; k) = \frac{1}{\Gamma(k/2)} 2^{1 - k/2} x^{k-1} e^{-x^2/2}, \quad x > 0

.. code-block:: julia

    Chi(k)       # Chi distribution with k degrees of freedom

.. _chisquare:

Chi-square Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

The `Chi square distribution <http://en.wikipedia.org/wiki/Chi-squared_distribution>`_ with k degrees of freedom is the distribution of the sume of sqaures of k independent variables that are normally distributed. The probability density function is:

.. math::

    f(x; k) = \frac{x^{k/2 - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0

.. code-block:: julia

    Chisq(k)      # Chi-squared distribution with k degrees of freedom

.. _erlang:

Erlang Distribution
~~~~~~~~~~~~~~~~~~~~

The probability density function of an `Erlang distribution <http://en.wikipedia.org/wiki/Erlang_distribution>`_ with shape parameter k and scale θ is

.. math::

    f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\Gamma(k) \theta^k}, \quad x > 0

.. code-block:: julia

    Erlang()       # Erlang distribution with unit shape and unit scale
    Erlang(k)      # Erlang distribution with shape parameter k and unit scale
    Erlang(k, s)   # Erlang distribution with shape parameter k and scale s

**Note:** The Erlang distribution is a special case of the Gamma distribution with integer shape parameter. 

.. _exponential:

Exponential Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of an `Exponential distribution <http://en.wikipedia.org/wiki/Exponential_distribution>`_ with scale θ is

.. math::

    f(x; \theta) = \frac{1}{\theta} e^{-\frac{x}{\theta}}, \quad x > 0

.. code-block:: julia

    Exponential()      # Exponential distribution with unit scale
    Exponential(s)     # Exponential distribution with scale s

.. _fdist:

F Distribution
~~~~~~~~~~~~~~~

The probability density function of an `F distribution <http://en.wikipedia.org/wiki/F-distribution>`_ with parameters d1 and d2 is

.. math::

    f(x; d_1, d_2) = \frac{1}{x B(d_1/2, d_2/2)} 
    \sqrt{\frac{(d_1 x)^{d_1} \cdot d_2^{d_2}}{(d_1 x + d_2)^{d_1 + d_2}}}

.. code-block:: julia

    FDist(d1, d2)     # F-Distribution with parameters d1 and d2    


.. _frechet:

Fréchet Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Fréchet distribution <http://en.wikipedia.org/wiki/Fréchet_distribution>`_ with shape k>0 and scale θ>0 is 

.. math::

    f(x; k, \theta) = \frac{k}{\theta} \left( \frac{x}{\theta} \right)^{-k-1} e^{-(x/\theta)^{-k}},
    \quad x > 0

.. code-block:: julia

    Frechet(k)       # Fréchet distribution with shape k and unit scale
    Frechet(k, s)    # Fréchet distribution with shape k and scale s


.. _gamma:

Gamma Distribution
~~~~~~~~~~~~~~~~~~~

The probability density function of a `Gamma distribution <http://en.wikipedia.org/wiki/Gamma_distribution>`_ with shape parameter α and scale θ is

.. math::

    f(x; \alpha, \theta) = \frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha}, 
    \quad x > 0

.. code-block:: julia

    Gamma()          # Gamma distribution with unit shape and unit scale
    Gamma(a)         # Gamma distribution with shape a and unit scale
    Gamma(a, s)      # Gamma distribution with shape a and scale s

.. _gumbel:

Gumbel Distribution
~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Gumbel distribution <http://en.wikipedia.org/wiki/Gumbel_distribution>`_ with location μ and scale θ is

.. math::

    f(x; \mu, \theta) = \frac{1}{\theta} e^{-(z + e^z)}, 
    \quad \text{ with } z = \frac{x - \mu}{\theta}

.. code-block:: julia

    Gumbel()          # Gumbel distribution with zero location and unit scale
    Gumbel(mu, s)     # Gumbel distribution with location mu and scale s


.. _inversegamma:

Inverse Gamma Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of an `inverse Gamma distribution <http://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_ with shape parameter α and scale θ is

.. math::

    f(x; \alpha, \theta) = \frac{\theta^\alpha x^{-(\alpha + 1)}}{\Gamma(\alpha)}
    e^{-\frac{\theta}{x}}, \quad x > 0

.. code-block:: julia

    InverseGamma(a, s)    # Inverted Gamma distribution with shape a and scale s


.. _inversegaussian:

Inverse Gaussian Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The probability density function of an `inverse Gaussian distribution <http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution>`_ with mean μ and shape λ is

.. math::

    f(x; \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}}
    \exp\!\left(\frac{-\lambda(x-\mu)^2}{2\mu^2x}\right), \quad x > 0

.. code-block:: julia
  
    InverseGaussian()              # Inverse Gaussian distribution with unit mean and unit shape    
    InverseGaussian(mu, lambda)    # Inverse Gaussian distribution with mean mu and shape lambda

.. _laplace:

Laplace Distribution
~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Laplace distribution <http://en.wikipedia.org/wiki/Laplace_distribution>`_ with location μ and scale θ is 

.. math::

    f(x; \mu, \theta) = \frac{1}{2 \theta} \exp \left(- \frac{|x - \mu|}{\theta} \right)

.. code-block:: julia

    Laplace()       # Laplace distribution with zero location and unit scale
    Laplace(u)      # Laplace distribution with location u and unit scale
    Laplace(u, s)   # Laplace distribution with location u ans scale s


.. _levy:

Lévy Distribution
~~~~~~~~~~~~~~~~~~

The probability density function os a `Lévy distribution <http://en.wikipedia.org/wiki/Lévy_distribution>`_ with location μ and scale θ is 

.. math::

    f(x; \mu, \theta) = \sqrt{\frac{\theta}{2 \pi (x - \mu)^3}}
    \exp \left( - \frac{\theta}{2 (x - \mu)} \right), \quad x > \mu

.. code-block:: julia

    Levy()         # Levy distribution with zero location and unit scale
    Levy(u)        # Levy distribution with location u and unit scale
    Levy(u, s)     # Levy distribution with location u ans scale s    


.. _logistic:

Logistic Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Logistic distribution <http://en.wikipedia.org/wiki/Logistic_distribution>`_ with location μ and scale θ is

.. math:: 

    f(x; \mu, \theta) = \frac{1}{4 \theta} \mathrm{sech}^2 
    \left( \frac{x - \mu}{\theta} \right)

.. code-block:: julia

    Logistic()       # Logistic distribution with zero location and unit scale
    Logistic(u)      # Logistic distribution with location u and unit scale
    Logistic(u, s)   # Logistic distribution with location u ans scale s     


.. _lognormal:

Log-normal Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

Let ``Z`` be a random variable of standard normal distribution, then the distribution of ``exp(μ + σZ)`` is a `Lognormal distribution <http://en.wikipedia.org/wiki/Log-normal_distribution>`_. The probability density function is 

.. math::

    f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}} 
    \exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right)

.. code-block:: julia

    LogNormal()          # Log-normal distribution with zero log-mean and unit scale
    LogNormal(mu)        # Log-normal distribution with log-mean mu and unit scale
    LogNormal(mu, sig)   # Log-normal distribution with log-mean mu and scale sig 


.. _normal:

Normal Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability density distribution of a `Normal distribution <http://en.wikipedia.org/wiki/Normal_distribution>`_ with mean μ and standard deviation σ is

.. math::

    f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
    \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)

.. code-block:: julia

    Normal()          # standard Normal distribution with zero mean and unit variance
    Normal(mu)        # Normal distribution with mean mu and unit variance
    Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2


.. _pareto:

Pareto Distribution
~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Pareto distribution <http://en.wikipedia.org/wiki/Pareto_distribution>`_ with scale θ and shape α is

.. math::

    f(x; \theta, \alpha) = \frac{\alpha \theta^\alpha}{x^{\alpha + 1}}, \quad x \ge \theta

.. code-block:: julia

    Pareto()          # Pareto distribution with unit scale and unit shape
    Pareto(sh, sc)    # Pareto distribution with shape sh and scale sc


.. _rayleigh:

Rayleigh Distribution
~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Rayleigh distribution <http://en.wikipedia.org/wiki/Rayleigh_distribution>`_ with scale σ is 

.. math::

    f(x; \sigma) = \frac{x}{\sigma^2} e^{-\frac{x^2}{2 \sigma^2}}

.. code-block:: julia

    Rayleigh()       # Rayleigh distribution with unit scale
    Rayleigh(s)      # Rayleigh distribution with scale s


.. _tdist:

(Student's) T-Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Students T distribution <http://en.wikipedia.org/wiki/T-distribution>`_ with d degrees of freedom is

.. math::

    f(x; d) = \frac{1}{\sqrt{d} B(1/2, d/2)}
    \left( 1 + \frac{x^2}{d} \right)^{-\frac{d + 1}{2}}

.. code-block:: julia

    TDist(d)      # t-distribution with d degrees of freedom

.. _triangular:

Triangular Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Triangular distribution <http://en.wikipedia.org/wiki/Triangular_distribution>`_ with lower limit a, upper limit b and mode c is

.. math::
    
    f(x; a, b, c)= \begin{cases}
        0 & \mathrm{for\ } x < a, \\
        \frac{2(x-a)}{(b-a)(c-a)} & \mathrm{for\ } a \le x \leq c, \\[4pt]
        \frac{2(b-x)}{(b-a)(b-c)} & \mathrm{for\ } c < x \le b, \\[4pt]
        0 & \mathrm{for\ } b < x,
        \end{cases}

.. code-block:: julia

    TriangularDist(a, b, c)     # Triangular distribution with lower limit a, upper limit b and mode c

.. _uniform:

Uniform Distribution
~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Continuous Uniform distribution <http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_ over an interval ``[a, b]`` is

.. math::

    f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b

.. code-block:: julia

    Uniform()        # Uniform distribution over [0, 1]
    Uniform(a, b)    # Uniform distribution over [a, b]

.. _vonmises:

Von Mises Distribution
~~~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `von Mises distribution <http://en.wikipedia.org/wiki/Von_Mises_distribution>`_ with mean μ and concentration κ is

.. math::

    f(x; \mu, \kappa) = \frac{1}{2 \pi I_0(\kappa)} \exp \left( \kappa \cos (x - \mu) \right)

.. code-block:: julia

    VonMises()       # von Mises distribution with zero mean and unit concentration
    VonMises(κ)      # von Mises distribution with zero mean and concentration κ
    VonMises(μ, κ)   # von Mises distribution with mean μ and concentration κ


.. _weibull:

Weibull Distribution
~~~~~~~~~~~~~~~~~~~~~

The probability density function of a `Weibull distribution <http://en.wikipedia.org/wiki/Weibull_distribution>`_ with shape k and scale θ is 

.. math::

    f(x; k, \theta) = \frac{k}{\theta} \left( \frac{x}{\theta} \right)^{k-1} e^{-(x/\theta)^k},
    \quad x \ge 0

.. code-block:: julia

    Weibull(k)       # Weibull distribution with shape k and unit scale
    Weibull(k, s)    # Weibull distribution with shape k and scale s     


