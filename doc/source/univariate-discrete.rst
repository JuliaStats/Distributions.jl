.. _univariate-discrete:

Univariate Discrete Distributions
----------------------------------------------------

.. DO NOT EDIT: this file is generated from Julia source.

.. function:: Bernoulli(p)

   A *Bernoulli distribution* is parameterized by a success rate ``p``\ , which takes value 1 with probability ``p`` and 0 with probability ``1-p``\ .

   .. math::

       P(X = k) = \begin{cases}
       1 - p & \quad \text{for } k = 0, \\
       p & \quad \text{for } k = 1.
       \end{cases}

   .. code-block:: julia

       Bernoulli()    # Bernoulli distribution with p = 0.5
       Bernoulli(p)   # Bernoulli distribution with success rate p

       params(d)      # Get the parameters, i.e. (p,)
       succprob(d)    # Get the success rate, i.e. p
       failprob(d)    # Get the failure rate, i.e. 1 - p

   External links:

   * `Bernoulli distribution on Wikipedia <http://en.wikipedia.org/wiki/Bernoulli_distribution>`_

.. function:: BetaBinomial(n,α,β)

   A *Beta-binomial distribution* is the compound distribution of the :func:`Binomial` distribution where the probability of success ``p`` is distributed according to the :func:`Beta`. It has three parameters: ``n``\ , the number of trials and two shape parameters ``α``\ , ``β``

   .. math::

       P(X = k) = {n \choose k} B(k + \alpha, n - k + \beta) / B(\alpha, \beta),  \quad \text{ for } k = 0,1,2, \ldots, n.

   .. code-block:: julia

       BetaBinomial(n, a, b)      # BetaBinomial distribution with n trials and shape parameters a, b

       params(d)       # Get the parameters, i.e. (n, a, b)
       ntrials(d)      # Get the number of trials, i.e. n

   External links:

   * `Beta-binomial distribution on Wikipedia <https://en.wikipedia.org/wiki/Beta-binomial_distribution>`_

.. function:: Binomial(n,p)

   A *Binomial distribution* characterizes the number of successes in a sequence of independent trials. It has two parameters: ``n``\ , the number of trials, and ``p``\ , the probability of success in an individual trial, with the distribution:

   .. math::

       P(X = k) = {n \choose k}p^k(1-p)^{n-k},  \quad \text{ for } k = 0,1,2, \ldots, n.

   .. code-block:: julia

       Binomial()      # Binomial distribution with n = 1 and p = 0.5
       Binomial(n)     # Binomial distribution for n trials with success rate p = 0.5
       Binomial(n, p)  # Binomial distribution for n trials with success rate p

       params(d)       # Get the parameters, i.e. (n, p)
       ntrials(d)      # Get the number of trials, i.e. n
       succprob(d)     # Get the success rate, i.e. p
       failprob(d)     # Get the failure rate, i.e. 1 - p

   External links:

   * `Binomial distribution on Wikipedia <http://en.wikipedia.org/wiki/Binomial_distribution>`_

.. function:: Categorical(p)

   A *Categorical distribution* is parameterized by a probability vector ``p`` (of length ``K``\ ).

   .. math::

       P(X = k) = p[k]  \quad \text{for } k = 1, 2, \ldots, K.

   .. code-block:: julia

       Categorical(p)   # Categorical distribution with probability vector p

       params(d)        # Get the parameters, i.e. (p,)
       probs(d)         # Get the probability vector, i.e. p
       ncategories(d)   # Get the number of categories, i.e. K

   Here, ``p`` must be a real vector, of which all components are nonnegative and sum to one.

   **Note:** The input vector ``p`` is directly used as a field of the constructed distribution, without being copied.

   External links:

   * `Categorical distribution on Wikipedia <http://en.wikipedia.org/wiki/Categorical_distribution>`_

.. function:: DiscreteUniform(a,b)

   A *Discrete uniform distribution* is a uniform distribution over a consecutive sequence of integers between ``a`` and ``b``\ , inclusive.

   .. math::

       P(X = k) = 1 / (b - a + 1) \quad \text{for } k = a, a+1, \ldots, b.

   .. code-block:: julia

       DiscreteUniform(a, b)   # a uniform distribution over {a, a+1, ..., b}

       params(d)       # Get the parameters, i.e. (a, b)
       span(d)         # Get the span of the support, i.e. (b - a + 1)
       probval(d)      # Get the probability value, i.e. 1 / (b - a + 1)
       minimum(d)      # Return a
       maximum(d)      # Return b

   External links

   * `Discrete uniform distribution on Wikipedia <http://en.wikipedia.org/wiki/Uniform_distribution_(discrete)>`_

.. function:: Geometric(p)

   A *Geometric distribution* characterizes the number of failures before the first success in a sequence of independent Bernoulli trials with success rate ``p``\ .

   .. math::

       P(X = k) = p (1 - p)^k, \quad \text{for } k = 0, 1, 2, \ldots.

   .. code-block:: julia

       Geometric()    # Geometric distribution with success rate 0.5
       Geometric(p)   # Geometric distribution with success rate p

       params(d)      # Get the parameters, i.e. (p,)
       succprob(d)    # Get the success rate, i.e. p
       failprob(d)    # Get the failure rate, i.e. 1 - p

   External links

   *  `Geometric distribution on Wikipedia <http://en.wikipedia.org/wiki/Geometric_distribution>`_

.. function:: Hypergeometric(s, f, n)

   A *Hypergeometric distribution* describes the number of successes in ``n`` draws without replacement from a finite population containing ``s`` successes and ``f`` failures.

   .. math::

       P(X = k) = {{{s \choose k} {f \choose {n-k}}}\over {s+f \choose n}}, \quad \text{for } k = \max(0, n - f), \ldots, \min(n, s).

   .. code-block:: julia

       Hypergeometric(s, f, n)  # Hypergeometric distribution for a population with
                                # s successes and f failures, and a sequence of n trials.

       params(d)       # Get the parameters, i.e. (s, f, n)

   External links

   * `Hypergeometric distribution on Wikipedia <http://en.wikipedia.org/wiki/Hypergeometric_distribution>`_

.. function:: NegativeBinomial(r,p)

   A *Negative binomial distribution* describes the number of failures before the ``r``\ th success in a sequence of independent Bernoulli trials. It is parameterized by ``r``\ , the number of successes, and ``p``\ , the probability of success in an individual trial.

   .. math::

       P(X = k) = {k + r - 1 \choose k} p^r (1 - p)^k, \quad \text{for } k = 0,1,2,\ldots.

   The distribution remains well-defined for any positive ``r``\ , in which case

   .. math::

       P(X = k) = \frac{\Gamma(k+r)}{k! \Gamma(r)} p^r (1 - p)^k, \quad \text{for } k = 0,1,2,\ldots.

   .. code-block:: julia

       NegativeBinomial()        # Negative binomial distribution with r = 1 and p = 0.5
       NegativeBinomial(r, p)    # Negative binomial distribution with r successes and success rate p

       params(d)       # Get the parameters, i.e. (r, p)
       succprob(d)     # Get the success rate, i.e. p
       failprob(d)     # Get the failure rate, i.e. 1 - p

   External links:

   * `Negative binomial distribution on Wikipedia <http://en.wikipedia.org/wiki/Negative_binomial_distribution>`_

.. function:: Poisson(λ)

   A *Poisson distribution* descibes the number of independent events occurring within a unit time interval, given the average rate of occurrence ``λ``\ .

   .. math::

       P(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}, \quad \text{ for } k = 0,1,2,\ldots.

   .. code-block:: julia

       Poisson()        # Poisson distribution with rate parameter 1
       Poisson(lambda)       # Poisson distribution with rate parameter lambda

       params(d)        # Get the parameters, i.e. (λ,)
       mean(d)          # Get the mean arrival rate, i.e. λ

   External links:

   * `Poisson distribution on Wikipedia <http://en.wikipedia.org/wiki/Poisson_distribution>`_

.. function:: PoissonBinomial(p)

   A *Poisson-binomial distribution* describes the number of successes in a sequence of independent trials, wherein each trial has a different success rate. It is parameterized by a vector ``p`` (of length ``K``\ ), where ``K`` is the total number of trials and ``p[i]`` corresponds to the probability of success of the ``i``\ th trial.

   .. math::

       P(X = k) = \sum\limits_{A\in F_k} \prod\limits_{i\in A} p[i] \prod\limits_{j\in A^c} (1-p[j]), \quad \text{ for } k = 0,1,2,\ldots,K,

   where :math:`F_k` is the set of all subsets of :math:`k` integers that can be selected from :math:`\{1,2,3,...,K\}`\ .

   .. code-block:: julia

       PoissonBinomial(p)   # Poisson Binomial distribution with success rate vector p

       params(d)            # Get the parameters, i.e. (p,)
       succprob(d)          # Get the vector of success rates, i.e. p
       failprob(d)          # Get the vector of failure rates, i.e. 1-p

   External links:

   * `Poisson-binomial distribution on Wikipedia <http://en.wikipedia.org/wiki/Poisson_binomial_distribution>`_

.. function:: Skellam(μ1, μ2)

   A *Skellam distribution* describes the difference between two independent :func:`Poisson` variables, respectively with rate ``μ1`` and ``μ2``\ .

   .. math::

       P(X = k) = e^{-(\mu_1 + \mu_2)} \left( \frac{\mu_1}{\mu_2} \right)^{k/2} I_k(2 \sqrt{\mu_1 \mu_2}) \quad \text{for integer } k

   where :math:`I_k` is the modified Bessel function of the first kind.

   .. code-block:: julia

       Skellam(mu1, mu2)   # Skellam distribution for the difference between two Poisson variables,
                           # respectively with expected values mu1 and mu2.

       params(d)           # Get the parameters, i.e. (mu1, mu2)

   External links:

   * `Skellam distribution on Wikipedia <http://en.wikipedia.org/wiki/Skellam_distribution>`_

