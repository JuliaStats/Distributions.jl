# Distribution Fitting

This package provides methods to fit a distribution to a given set of samples. Generally, one may write

```julia
d = fit(D, x)
```

This statement fits a distribution of type `D` to a given dataset `x`, where `x` should be an array comprised of all samples. The fit function will choose a reasonable way to fit the distribution, which, in most cases, is [maximum likelihood estimation](http://en.wikipedia.org/wiki/Maximum_likelihood).

!!! note

    One can use as the first argument simply the distribution name, like `Binomial`,
    or a concrete distribution with a type parameter, like `Normal{Float64}` or
    `Exponential{Float32}`.  However, in the latter case the type parameter of
    the distribution will be ignored:

    ```julia
    julia> fit(Cauchy{Float32}, collect(-4:4))
    Cauchy{Float64}(μ=0.0, σ=2.0)
    ```

## Maximum Likelihood Estimation

The function `fit_mle` is for maximum likelihood estimation.

### Synopsis

```@docs
fit(D, x)
fit(D, x, w)
fit_mle(D, x)
fit_mle(D, x, w)
```

### Applicable distributions

The `fit_mle` method has been implemented for the following distributions:

**Univariate:**

- [`Bernoulli`](@ref)
- [`Beta`](@ref)
- [`Binomial`](@ref)
- [`Categorical`](@ref)
- [`DiscreteUniform`](@ref)
- [`Exponential`](@ref)
- [`LogNormal`](@ref)
- [`Normal`](@ref)
- [`Gamma`](@ref)
- [`Geometric`](@ref)
- [`Laplace`](@ref)
- [`Pareto`](@ref)
- [`Poisson`](@ref)
- [`Rayleigh`](@ref)
- [`InverseGaussian`](@ref)
- [`Uniform`](@ref)
- [`Weibull`](@ref)

**Multivariate:**

- [`Multinomial`](@ref)
- [`MvNormal`](@ref)
- [`Dirichlet`](@ref)

For most of these distributions, the usage is as described above. For a few special distributions that require additional information for estimation, we have to use a modified interface:

```julia
fit_mle(Binomial, n, x)        # n is the number of trials in each experiment
fit_mle(Binomial, n, x, w)

fit_mle(Categorical, k, x)     # k is the space size (i.e. the number of distinct values)
fit_mle(Categorical, k, x, w)

fit_mle(Categorical, x)        # equivalent to fit_mle(Categorical, max(x), x)
fit_mle(Categorical, x, w)
```

## Sufficient Statistics

For many distributions, the estimation can be based on (sum of) sufficient statistics computed from a dataset. To simplify implementation, for such distributions, we implement `suffstats` method instead of `fit_mle` directly:

```julia
ss = suffstats(D, x)        # ss captures the sufficient statistics of x
ss = suffstats(D, x, w)     # ss captures the sufficient statistics of a weighted dataset

d = fit_mle(D, ss)          # maximum likelihood estimation based on sufficient stats
```

When `fit_mle` on `D` is invoked, a fallback `fit_mle` method will first call `suffstats` to compute the sufficient statistics, and then a `fit_mle` method on sufficient statistics to get the result. For some distributions, this way is not the most efficient, and we specialize the `fit_mle` method to implement more efficient estimation algorithms.


## Maximum-a-Posteriori Estimation

Maximum-a-Posteriori (MAP) estimation is also supported by this package, which is implemented as part of the conjugate exponential family framework (see :ref:`Conjugate Prior and Posterior <ref-conj>`).
