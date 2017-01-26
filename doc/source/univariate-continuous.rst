.. _univariate-continuous:

Univariate Continuous Distributions
----------------------------------------------------

.. DO NOT EDIT: this file is generated from Julia source.

.. function:: Arcsine(a,b)

   The *Arcsine distribution* has probability density function

   .. math::

       f(x) = \frac{1}{\pi \sqrt{(x - a) (b - x)}}, \quad x \in [a, b]

   .. code-block:: julia

       Arcsine()        # Arcsine distribution with support [0, 1]
       Arcsine(b)       # Arcsine distribution with support [0, b]
       Arcsine(a, b)    # Arcsine distribution with support [a, b]

       params(d)        # Get the parameters, i.e. (a, b)
       minimum(d)       # Get the lower bound, i.e. a
       maximum(d)       # Get the upper bound, i.e. b
       location(d)      # Get the left bound, i.e. a
       scale(d)         # Get the span of the support, i.e. b - a

   External links

   * `Arcsine distribution on Wikipedia <http://en.wikipedia.org/wiki/Arcsine_distribution>`_

.. function:: Beta(α,β)

   The *Beta distribution* has probability density function

   .. math::

       f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}
        x^{\alpha - 1} (1 - x)^{\beta - 1}, \quad x \in [0, 1]

   The Beta distribution is related to the :func:`Gamma` distribution via the property that if :math:`X \sim \operatorname{Gamma}(\alpha)` and :math:`Y \sim \operatorname{Gamma} (\beta)` independently, then :math:`X / (X + Y) \sim \operatorname{Beta}(\alpha, \beta)`\ .

   .. code-block:: julia

       Beta()        # equivalent to Beta(1.0, 1.0)
       Beta(a)       # equivalent to Beta(a, a)
       Beta(a, b)    # Beta distribution with shape parameters a and b

       params(d)     # Get the parameters, i.e. (a, b)

   External links

   * `Beta distribution on Wikipedia <http://en.wikipedia.org/wiki/Beta_distribution>`_

.. function:: BetaPrime(α,β)

   The *Beta prime distribution* has probability density function

   .. math::

       f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)}
       x^{\alpha - 1} (1 + x)^{- (\alpha + \beta)}, \quad x > 0

   The Beta prime distribution is related to the :func:`Beta` distribution via the relation ship that if :math:`X \sim \operatorname{Beta}(\alpha, \beta)` then :math:`\frac{X}{1 - X} \sim \operatorname{BetaPrime}(\alpha, \beta)`

   .. code-block:: julia

       BetaPrime()        # equivalent to BetaPrime(0.0, 1.0)
       BetaPrime(a)       # equivalent to BetaPrime(a, a)
       BetaPrime(a, b)    # Beta prime distribution with shape parameters a and b

       params(d)          # Get the parameters, i.e. (a, b)

   External links

   * `Beta prime distribution on Wikipedia <http://en.wikipedia.org/wiki/Beta_prime_distribution>`_



.. function:: Cauchy(μ, σ)

   The *Cauchy distribution* with location ``μ`` and scale ``σ`` has probability density function

   .. math::

       f(x; \mu, \sigma) = \frac{1}{\pi \sigma \left(1 + \left(\frac{x - \mu}{\sigma} \right)^2 \right)}

   .. code-block:: julia

       Cauchy()         # Standard Cauchy distribution, i.e. Cauchy(0.0, 1.0)
       Cauchy(u)        # Cauchy distribution with location u and unit scale, i.e. Cauchy(u, 1.0)
       Cauchy(u, b)     # Cauchy distribution with location u and scale b

       params(d)        # Get the parameters, i.e. (u, b)
       location(d)      # Get the location parameter, i.e. u
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Cauchy distribution on Wikipedia <http://en.wikipedia.org/wiki/Cauchy_distribution>`_

.. function:: Chi(ν)

   The *Chi distribution* ``ν`` degrees of freedom has probability density function

   .. math::

       f(x; k) = \frac{1}{\Gamma(k/2)} 2^{1 - k/2} x^{k-1} e^{-x^2/2}, \quad x > 0

   It is the distribution of the square-root of a :func:`Chisq` variate.

   .. code-block:: julia

       Chi(k)       # Chi distribution with k degrees of freedom

       params(d)    # Get the parameters, i.e. (k,)
       dof(d)       # Get the degrees of freedom, i.e. k

   External links

   * `Chi distribution on Wikipedia <http://en.wikipedia.org/wiki/Chi_distribution>`_

.. function:: Chisq(ν)

   The *Chi squared distribution* (typically written χ²) with ``ν`` degrees of freedom has the probability density function

   .. math::

       f(x; k) = \frac{x^{k/2 - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0.

   If ``ν`` is an integer, then it is the distribution of the sum of squares of ``ν`` independent standard :func:`Normal` variates.

   .. code-block:: julia

       Chisq(k)     # Chi-squared distribution with k degrees of freedom

       params(d)    # Get the parameters, i.e. (k,)
       dof(d)       # Get the degrees of freedom, i.e. k

   External links

   * `Chi-squared distribution on Wikipedia <http://en.wikipedia.org/wiki/Chi-squared_distribution>`_

.. function:: Erlang(α,θ)

   The *Erlang distribution* is a special case of a :func:`Gamma` distribution with integer shape parameter.

   .. code-block:: julia

       Erlang()       # Erlang distribution with unit shape and unit scale, i.e. Erlang(1.0, 1.0)
       Erlang(a)      # Erlang distribution with shape parameter a and unit scale, i.e. Erlang(a, 1.0)
       Erlang(a, s)   # Erlang distribution with shape parameter a and scale b

   External links

   * `Erlang distribution on Wikipedia <http://en.wikipedia.org/wiki/Erlang_distribution>`_

.. function:: Exponential(θ)

   The *Exponential distribution* with scale parameter ``θ`` has probability density function

   .. math::

       f(x; \theta) = \frac{1}{\theta} e^{-\frac{x}{\theta}}, \quad x > 0

   .. code-block:: julia

       Exponential()      # Exponential distribution with unit scale, i.e. Exponential(1.0)
       Exponential(b)     # Exponential distribution with scale b

       params(d)          # Get the parameters, i.e. (b,)
       scale(d)           # Get the scale parameter, i.e. b
       rate(d)            # Get the rate parameter, i.e. 1 / b

   External links

   * `Exponential distribution on Wikipedia <http://en.wikipedia.org/wiki/Exponential_distribution>`_

.. function:: FDist(ν1,ν2)

   The *F distribution* has probability density function

   .. math::

       f(x; \nu_1, \nu_2) = \frac{1}{x B(\nu_1/2, \nu_2/2)}
       \sqrt{\frac{(\nu_1 x)^{\nu_1} \cdot \nu_2^{\nu_2}}{(\nu_1 x + \nu_2)^{\nu_1 + \nu_2}}},
       \quad x>0

   It is related to the :func:`Chisq` distribution via the property that if :math:`X_1 \sim \operatorname{Chisq}(\nu_1)` and :math:`X_2 \sim \operatorname{Chisq}(\nu_2)`\ , then  $(X_1/\\nu_1) / (X_2 / \\nu_2) \\sim FDist(\\nu_1, \\nu_2)`.

   .. code-block:: julia

       FDist(d1, d2)     # F-Distribution with parameters d1 and d2

       params(d)         # Get the parameters, i.e. (d1, d2)

   External links

   * `F distribution on Wikipedia <http://en.wikipedia.org/wiki/F-distribution>`_     

.. function:: Frechet(α,θ)

   The *Fréchet distribution* with shape ``α`` and scale ``θ`` has probability density function

   .. math::

       f(x; \alpha, \theta) = \frac{\alpha}{\theta} \left( \frac{x}{\theta} \right)^{-\alpha-1} 
       e^{-(x/\theta)^{-\alpha}}, \quad x > 0

   .. code-block:: julia

       Frechet()        # Fréchet distribution with unit shape and unit scale, i.e. Frechet(1.0, 1.0)
       Frechet(a)       # Fréchet distribution with shape a and unit scale, i.e. Frechet(a, 1.0)
       Frechet(a, b)    # Fréchet distribution with shape a and scale b

       params(d)        # Get the parameters, i.e. (a, b)
       shape(d)         # Get the shape parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Fréchet_distribution on Wikipedia <http://en.wikipedia.org/wiki/Fréchet_distribution>`_

.. function:: Gamma(α,θ)

   The *Gamma distribution* with shape parameter ``α`` and scale ``θ`` has probability density function

   .. math::

       f(x; \alpha, \theta) = \frac{x^{\alpha-1} e^{-x/\theta}}{\Gamma(\alpha) \theta^\alpha},
       \quad x > 0

   .. code-block:: julia

       Gamma()          # Gamma distribution with unit shape and unit scale, i.e. Gamma(1.0, 1.0)
       Gamma(a)         # Gamma distribution with shape a and unit scale, i.e. Gamma(a, 1.0)
       Gamma(a, b)      # Gamma distribution with shape a and scale b

       params(d)        # Get the parameters, i.e. (a, b)
       shape(d)         # Get the shape parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Gamma distribution on Wikipedia <http://en.wikipedia.org/wiki/Gamma_distribution>`_

.. function:: GeneralizedExtremeValue(μ, σ, ξ)

   The *Generalized extreme value distribution* with shape parameter ``ξ``\ , scale ``σ`` and location ``μ`` has probability density function

   .. math::

       f(x; \xi, \sigma, \mu) = \begin{cases}
               \frac{1}{\sigma} \left[ 1+\left(\frac{x-\mu}{\sigma}\right)\xi\right]^{-1/\xi-1} \exp\left\{-\left[ 1+ \left(\frac{x-\mu}{\sigma}\right)\xi\right]^{-1/\xi} \right\} & \text{for } \xi \neq 0 \\
               \frac{1}{\sigma} \exp\left\{-\frac{x-\mu}{\sigma}\right\} \exp\left\{-\exp\left[-\frac{x-\mu}{\sigma}\right]\right\} & \text{for } \xi = 0
           \end{cases}

   for

   .. math::

       x \in \begin{cases}
               \left[ \mu - \frac{\sigma}{\xi}, + \infty \right) & \text{for } \xi > 0 \\
               \left( - \infty, + \infty \right) & \text{for } \xi = 0 \\
               \left( - \infty, \mu - \frac{\sigma}{\xi} \right] & \text{for } \xi < 0
           \end{cases}

   .. code-block:: julia

       GeneralizedExtremeValue(k, s, m)      # Generalized Pareto distribution with shape k, scale s and location m.

       params(d)       # Get the parameters, i.e. (k, s, m)
       shape(d)        # Get the shape parameter, i.e. k (sometimes called c)
       scale(d)        # Get the scale parameter, i.e. s
       location(d)     # Get the location parameter, i.e. m

   External links

   * `Generalized extreme value distribution on Wikipedia <https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution>`_

.. function:: GeneralizedPareto(ξ, σ, μ)

   The *Generalized Pareto distribution* with shape parameter ``ξ``\ , scale ``σ`` and location ``μ`` has probability density function

   .. math::

       f(x; \xi, \sigma, \mu) = \begin{cases}
               \frac{1}{\sigma}(1 + \xi \frac{x - \mu}{\sigma} )^{-\frac{1}{\xi} - 1} & \text{for } \xi \neq 0 \\
               \frac{1}{\sigma} e^{-\frac{\left( x - \mu \right) }{\sigma}} & \text{for } \xi = 0
           \end{cases}~,
           \quad x \in \begin{cases}
               \left[ \mu, \infty \right] & \text{for } \xi \geq 0 \\
               \left[ \mu, \mu - \sigma / \xi \right] & \text{for } \xi < 0
           \end{cases}

   .. code-block:: julia

       GeneralizedPareto()             # Generalized Pareto distribution with unit shape and unit scale, i.e. GeneralizedPareto(1.0, 1.0, 0.0)
       GeneralizedPareto(k, s)         # Generalized Pareto distribution with shape k and scale s, i.e. GeneralizedPareto(k, s, 0.0)
       GeneralizedPareto(k, s, m)      # Generalized Pareto distribution with shape k, scale s and location m.

       params(d)       # Get the parameters, i.e. (k, s, m)
       shape(d)        # Get the shape parameter, i.e. k
       scale(d)        # Get the scale parameter, i.e. s
       location(d)     # Get the location parameter, i.e. m

   External links

   * `Generalized Pareto distribution on Wikipedia <https://en.wikipedia.org/wiki/Generalized_Pareto_distribution>`_

.. function:: Gumbel(μ, θ)

   The *Gumbel distribution*  with location ``μ`` and scale ``θ`` has probability density function

   .. math::

       f(x; \mu, \theta) = \frac{1}{\theta} e^{-(z + e^z)},
       \quad \text{ with } z = \frac{x - \mu}{\theta}

   .. code-block:: julia

       Gumbel()            # Gumbel distribution with zero location and unit scale, i.e. Gumbel(0.0, 1.0)
       Gumbel(u)           # Gumbel distribution with location u and unit scale, i.e. Gumbel(u, 1.0)
       Gumbel(u, b)        # Gumbel distribution with location u and scale b

       params(d)        # Get the parameters, i.e. (u, b)
       location(d)      # Get the location parameter, i.e. u
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Gumbel distribution on Wikipedia <http://en.wikipedia.org/wiki/Gumbel_distribution>`_

.. function:: InverseGamma(α, θ)

   The *inverse gamma distribution* with shape parameter ``α`` and scale ``θ`` has probability density function

   .. math::

       f(x; \alpha, \theta) = \frac{\theta^\alpha x^{-(\alpha + 1)}}{\Gamma(\alpha)}
       e^{-\frac{\theta}{x}}, \quad x > 0

   It is related to the :func:`Gamma` distribution: if :math:`X \sim \operatorname{Gamma}(\alpha, \beta)`\ , then :math:`1 / X \sim \operatorname{InverseGamma}(\alpha, \beta^{-1})`\ .

   .. code-block:: julia

   .. code-block:: julia

       InverseGamma()        # Inverse Gamma distribution with unit shape and unit scale, i.e. InverseGamma(1.0, 1.0)
       InverseGamma(a)       # Inverse Gamma distribution with shape a and unit scale, i.e. InverseGamma(a, 1.0)
       InverseGamma(a, b)    # Inverse Gamma distribution with shape a and scale b

       params(d)        # Get the parameters, i.e. (a, b)
       shape(d)         # Get the shape parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Inverse gamma distribution on Wikipedia <http://en.wikipedia.org/wiki/Inverse-gamma_distribution>`_

.. function:: InverseGaussian(μ,λ)

   The *inverse Gaussian distribution* with mean ``μ`` and shape ``λ`` has probability density function

   .. math::

       f(x; \mu, \lambda) = \sqrt{\frac{\lambda}{2\pi x^3}}
       \exp\!\left(\frac{-\lambda(x-\mu)^2}{2\mu^2x}\right), \quad x > 0

   .. code-block:: julia

       InverseGaussian()              # Inverse Gaussian distribution with unit mean and unit shape, i.e. InverseGaussian(1.0, 1.0)
       InverseGaussian(mu),           # Inverse Gaussian distribution with mean mu and unit shape, i.e. InverseGaussian(u, 1.0)
       InverseGaussian(mu, lambda)    # Inverse Gaussian distribution with mean mu and shape lambda

       params(d)           # Get the parameters, i.e. (mu, lambda)
       mean(d)             # Get the mean parameter, i.e. mu
       shape(d)            # Get the shape parameter, i.e. lambda

   External links

   * `Inverse Gaussian distribution on Wikipedia <http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution>`_

.. function:: Laplace(μ,θ)

   The *Laplace distribution* with location ``μ`` and scale ``θ`` has probability density function

   .. math::

       f(x; \mu, \beta) = \frac{1}{2 \beta} \exp \left(- \frac{|x - \mu|}{\beta} \right)

   .. code-block:: julia

       Laplace()       # Laplace distribution with zero location and unit scale, i.e. Laplace(0.0, 1.0)
       Laplace(u)      # Laplace distribution with location u and unit scale, i.e. Laplace(u, 1.0)
       Laplace(u, b)   # Laplace distribution with location u ans scale b

       params(d)       # Get the parameters, i.e. (u, b)
       location(d)     # Get the location parameter, i.e. u
       scale(d)        # Get the scale parameter, i.e. b

   External links

   * `Laplace distribution on Wikipedia <http://en.wikipedia.org/wiki/Laplace_distribution>`_

.. function:: Levy(μ, σ)

   The *Lévy distribution* with location ``μ`` and scale ``σ`` has probability density function

   .. math::

       f(x; \mu, \sigma) = \sqrt{\frac{\sigma}{2 \pi (x - \mu)^3}}
       \exp \left( - \frac{\sigma}{2 (x - \mu)} \right), \quad x > \mu

   .. code-block:: julia

       Levy()         # Levy distribution with zero location and unit scale, i.e. Levy(0.0, 1.0)
       Levy(u)        # Levy distribution with location u and unit scale, i.e. Levy(u, 1.0)
       Levy(u, c)     # Levy distribution with location u ans scale c

       params(d)      # Get the parameters, i.e. (u, c)
       location(d)    # Get the location parameter, i.e. u

   External links

   * `Lévy distribution on Wikipedia <http://en.wikipedia.org/wiki/Lévy_distribution>`_

.. function:: LogNormal(μ,σ)

   The *log normal distribution* is the distribution of the exponential of a :func:`Normal` variate: if :math:`X \sim \operatorname{Normal}(\mu, \sigma)` then :math:`\exp(X) \sim \operatorname{LogNormal}(\mu,\sigma)`\ . The probability density function is

   .. math::

       f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}
       \exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),
       \quad x > 0

   .. code-block:: julia

       LogNormal()          # Log-normal distribution with zero log-mean and unit scale
       LogNormal(mu)        # Log-normal distribution with log-mean mu and unit scale
       LogNormal(mu, sig)   # Log-normal distribution with log-mean mu and scale sig

       params(d)            # Get the parameters, i.e. (mu, sig)
       meanlogx(d)          # Get the mean of log(X), i.e. mu
       varlogx(d)           # Get the variance of log(X), i.e. sig^2
       stdlogx(d)           # Get the standard deviation of log(X), i.e. sig

   External links

   * `Log normal distribution on Wikipedia <http://en.wikipedia.org/wiki/Log-normal_distribution>`_

.. function:: Logistic(μ,θ)

   The *Logistic distribution* with location ``μ`` and scale ``θ`` has probability density function

   .. math::

       f(x; \mu, \theta) = \frac{1}{4 \theta} \mathrm{sech}^2
       \left( \frac{x - \mu}{2 \theta} \right)

   .. code-block:: julia

       Logistic()       # Logistic distribution with zero location and unit scale, i.e. Logistic(0.0, 1.0)
       Logistic(u)      # Logistic distribution with location u and unit scale, i.e. Logistic(u, 1.0)
       Logistic(u, b)   # Logistic distribution with location u ans scale b

       params(d)       # Get the parameters, i.e. (u, b)
       location(d)     # Get the location parameter, i.e. u
       scale(d)        # Get the scale parameter, i.e. b

   External links

   * `Logistic distribution on Wikipedia <http://en.wikipedia.org/wiki/Logistic_distribution>`_

.. function:: Normal(μ,σ)

   The *Normal distribution* with mean ``μ`` and standard deviation ``σ`` has probability density function

   .. math::

       f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
       \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)

   .. code-block:: julia

       Normal()          # standard Normal distribution with zero mean and unit variance
       Normal(mu)        # Normal distribution with mean mu and unit variance
       Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2

       params(d)         # Get the parameters, i.e. (mu, sig)
       mean(d)           # Get the mean, i.e. mu
       std(d)            # Get the standard deviation, i.e. sig

   External links

   * `Normal distribution on Wikipedia <http://en.wikipedia.org/wiki/Normal_distribution>`_

.. function:: NormalInverseGaussian(μ,α,β,δ)

   The *Normal-inverse Gaussian distribution* with location ``μ``\ , tail heaviness ``α``\ , asymmetry parameter ``β`` and scale ``δ`` has probability density function

   .. math::

       f(x; \mu, \alpha, \beta, \delta) = \frac{\alpha\delta K_1 \left(\alpha\sqrt{\delta^2 + (x - \mu)^2}\right)}{\pi \sqrt{\delta^2 + (x - \mu)^2}} \; e^{\delta \gamma + \beta (x - \mu)}

   where :math:`K_j` denotes a modified Bessel function of the third kind.

   External links

   * `Normal-inverse Gaussian distribution on Wikipedia <http://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution>`_

.. function:: Pareto(α,θ)

   The *Pareto distribution* with shape ``α`` and scale ``θ`` has probability density function

   .. math::

       f(x; \alpha, \theta) = \frac{\alpha \theta^\alpha}{x^{\alpha + 1}}, \quad x \ge \theta

   .. code-block:: julia

       Pareto()            # Pareto distribution with unit shape and unit scale, i.e. Pareto(1.0, 1.0)
       Pareto(a)           # Pareto distribution with shape a and unit scale, i.e. Pareto(a, 1.0)
       Pareto(a, b)        # Pareto distribution with shape a and scale b

       params(d)        # Get the parameters, i.e. (a, b)
       shape(d)         # Get the shape parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b

   External links  * `Pareto distribution on Wikipedia <http://en.wikipedia.org/wiki/Pareto_distribution>`_

.. function:: Rayleigh(σ)

   The *Rayleigh distribution* with scale ``σ`` has probability density function

   .. math::

       f(x; \sigma) = \frac{x}{\sigma^2} e^{-\frac{x^2}{2 \sigma^2}}, \quad x > 0

   It is related to the :func:`Normal` distribution via the property that if :math:`X, Y \sim \operatorname{Normal}(0,\sigma)`\ , independently, then :math:`\sqrt{X^2 + Y^2} \sim \operatorname{Rayleigh}(\sigma)`\ .

   .. code-block:: julia

       Rayleigh()       # Rayleigh distribution with unit scale, i.e. Rayleigh(1.0)
       Rayleigh(s)      # Rayleigh distribution with scale s

       params(d)        # Get the parameters, i.e. (s,)
       scale(d)         # Get the scale parameter, i.e. s

   External links

   * `Rayleigh distribution on Wikipedia <http://en.wikipedia.org/wiki/Rayleigh_distribution>`_

.. function:: SymTriangularDist(μ,σ)

   The *Symmetric triangular distribution* with location ``μ`` and scale ``σ`` has probability density function

   .. math::

       f(x; \mu, \sigma) = \frac{1}{\sigma} \left( 1 - \left| \frac{x - \mu}{\sigma} \right| \right), \quad \mu - \sigma \le x \le \mu + \sigma

   .. code-block:: julia

       SymTriangularDist()         # Symmetric triangular distribution with zero location and unit scale
       SymTriangularDist(u)        # Symmetric triangular distribution with location u and unit scale
       SymTriangularDist(u, s)     # Symmetric triangular distribution with location u and scale s

       params(d)       # Get the parameters, i.e. (u, s)
       location(d)     # Get the location parameter, i.e. u
       scale(d)        # Get the scale parameter, i.e. s

.. function:: TDist(ν)

   The *Students T distribution* with ``ν`` degrees of freedom has probability density function

   .. math::

       f(x; d) = \frac{1}{\sqrt{d} B(1/2, d/2)}
       \left( 1 + \frac{x^2}{d} \right)^{-\frac{d + 1}{2}}

   .. code-block:: julia

       TDist(d)      # t-distribution with d degrees of freedom

       params(d)     # Get the parameters, i.e. (d,)
       dof(d)        # Get the degrees of freedom, i.e. d

   External links

   `Student's T distribution on Wikipedia <https://en.wikipedia.org/wiki/Student%27s_t-distribution>`_

.. function:: TriangularDist(a,b,c)

   The *triangular distribution* with lower limit ``a``\ , upper limit ``b`` and mode ``c`` has probability density function

   .. math::

       f(x; a, b, c)= \begin{cases}
               0 & \mathrm{for\ } x < a, \\
               \frac{2(x-a)}{(b-a)(c-a)} & \mathrm{for\ } a \le x \leq c, \\[4pt]
               \frac{2(b-x)}{(b-a)(b-c)} & \mathrm{for\ } c < x \le b, \\[4pt]
               0 & \mathrm{for\ } b < x,
               \end{cases}

   .. code-block:: julia

       TriangularDist(a, b)        # Triangular distribution with lower limit a, upper limit b, and mode (a+b)/2
       TriangularDist(a, b, c)     # Triangular distribution with lower limit a, upper limit b, and mode c

       params(d)       # Get the parameters, i.e. (a, b, c)
       minimum(d)      # Get the lower bound, i.e. a
       maximum(d)      # Get the upper bound, i.e. b
       mode(d)         # Get the mode, i.e. c

   External links

   * `Triangular distribution on Wikipedia <http://en.wikipedia.org/wiki/Triangular_distribution>`_

.. function:: Uniform(a,b)

   The *continuous uniform distribution* over an interval :math:`[a, b]` has probability density function

   .. math::

       f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b

   .. code-block:: julia

       Uniform()        # Uniform distribution over [0, 1]
       Uniform(a, b)    # Uniform distribution over [a, b]

       params(d)        # Get the parameters, i.e. (a, b)
       minimum(d)       # Get the lower bound, i.e. a
       maximum(d)       # Get the upper bound, i.e. b
       location(d)      # Get the location parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b - a

   External links

   * `Uniform distribution (continuous) on Wikipedia <http://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_

.. function:: VonMises(μ, κ)

   The *von Mises distribution* with mean ``μ`` and concentration ``κ`` has probability density function

   .. math::

       f(x; \mu, \kappa) = \frac{1}{2 \pi I_0(\kappa)} \exp \left( \kappa \cos (x - \mu) \right)

   .. code-block:: julia

       VonMises()       # von Mises distribution with zero mean and unit concentration
       VonMises(κ)      # von Mises distribution with zero mean and concentration κ
       VonMises(μ, κ)   # von Mises distribution with mean μ and concentration κ

   External links

   * `von Mises distribution on Wikipedia <http://en.wikipedia.org/wiki/Von_Mises_distribution>`_

.. function:: Weibull(α,θ)

   The *Weibull distribution* with shape ``α`` and scale ``θ`` has probability density function

   .. math::

       f(x; \alpha, \theta) = \frac{\alpha}{\theta} \left( \frac{x}{\theta} \right)^{\alpha-1} e^{-(x/\theta)^\alpha},
           \quad x \ge 0

   .. code-block:: julia

       Weibull()        # Weibull distribution with unit shape and unit scale, i.e. Weibull(1.0, 1.0)
       Weibull(a)       # Weibull distribution with shape a and unit scale, i.e. Weibull(a, 1.0)
       Weibull(a, b)    # Weibull distribution with shape a and scale b

       params(d)        # Get the parameters, i.e. (a, b)
       shape(d)         # Get the shape parameter, i.e. a
       scale(d)         # Get the scale parameter, i.e. b

   External links

   * `Weibull distribution on Wikipedia <http://en.wikipedia.org/wiki/Weibull_distribution>`_

