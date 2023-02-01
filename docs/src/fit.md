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
- [`ProductDistribution`](@ref)

For most of these distributions, the usage is as described above. For a few special distributions that require additional information for estimation, we have to use a modified interface:

```julia
fit_mle(Binomial, n, x)        # n is the number of trials in each experiment
fit_mle(Binomial, n, x, w)

fit_mle(Categorical, k, x)     # k is the space size (i.e. the number of distinct values)
fit_mle(Categorical, k, x, w)

fit_mle(Categorical, x)        # equivalent to fit_mle(Categorical, max(x), x)
fit_mle(Categorical, x, w)
```

It is also possible to directly input a distribution `fit_mle(d::Distribution, x[, w])`. This form avoids the extra arguments:

```julia
fit_mle(Binomial(n, 0.1), x) 
# equivalent to fit_mle(Binomial, ntrials(Binomial(n, 0.1)), x), here the parameter 0.1 is not used

fit_mle(Categorical(p), x) 
# equivalent to fit_mle(Categorical, ncategories(Categorical(p)), x), here the only the length of p is used not its values

d = product_distribution([Exponential(0.5), Normal(11.3, 3.2)])
fit_mle(d, x) 
# equivalent to product_distribution([fit_mle(Exponential, x[1,:]), fit_mle(Normal, x[2, :])]). Again parameters of d are not used.
```

Note that for standard distributions, the values of the distribution parameters `d` are not used in `fit_mle` only the “structure” of `d` is passed into `fit_mle`.
However, for complex Maximum Likelihood estimation requiring optimization, e.g., EM algorithm, one could use `D` as an initial guess.

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
