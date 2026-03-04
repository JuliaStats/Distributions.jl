# Getting Started

## Installation

The Distributions package is available through the Julia package system by running `Pkg.add("Distributions")`.
Throughout, we assume that you have installed the package.

## Starting With a Normal Distribution

We start by drawing 100 observations from a standard-normal random variable.

The first step is to set up the environment:

```jldoctest getting-started
julia> using Random, Distributions

julia> Random.seed!(123); # Setting the seed
```

Then, we create a standard-normal distribution `d` and obtain samples using `rand`:

```jldoctest getting-started
julia> d = Normal()
Normal{Float64}(μ=0.0, σ=1.0)
```

The object `d` represents a probability distribution, in our case the standard-normal distribution.
One can query its properties such as the mean:

```jldoctest getting-started
julia> mean(d)
0.0
```

We can also draw samples from `d` with `rand`.
```julia-repl
julia> x = rand(d, 100)
100-element Vector{Float64}:
  0.376264
 -0.405272
 ...
```

You can easily obtain the `pdf`, `cdf`, `quantile`, and many other functions for a distribution. For instance, the median (50th percentile) and the 95th percentile for the standard-normal distribution are given by:

```jldoctest getting-started; filter = r"(\d*)\.(\d{3})\d+" => s"\1.\2***"
julia> quantile.(Normal(), [0.5, 0.95])
2-element Vector{Float64}:
 0.0
 1.6448536269514717
```

The normal distribution is parameterized by its mean and standard deviation. To draw random samples from a normal distribution with mean 1 and standard deviation 2, you write:

```julia-repl
julia> rand(Normal(1, 2), 100)
```

## Using Other Distributions

The package contains a large number of additional distributions of three main types:

* `Univariate == ArrayLikeVariate{0}`
* `Multivariate == ArrayLikeVariate{1}`
* `Matrixvariate == ArrayLikeVariate{2}`

Each type splits further into `Discrete` and `Continuous`.

For instance, you can define the following distributions (among many others):

```julia-repl
julia> Binomial(n, p) # Discrete univariate
julia> Cauchy(u, b)  # Continuous univariate
julia> Multinomial(n, p) # Discrete multivariate
julia> Wishart(nu, S) # Continuous matrix-variate
```

In addition, you can create truncated distributions from univariate distributions:

```julia-repl
julia> truncated(Normal(mu, sigma), l, u)
```

To find out which parameters are appropriate for a given distribution `D`, you can use `fieldnames(D)`:

```jldoctest getting-started
julia> fieldnames(Cauchy)
(:μ, :σ)
```

This tells you that a Cauchy distribution is initialized with location `μ` and scale `β`.

## Estimate the Parameters

It is often useful to approximate an empirical distribution with a theoretical distribution. As an example, we can use the array `x` we created above and ask which normal distribution best describes it:

```julia-repl
julia> fit(Normal, x)
Normal{Float64}(μ=-0.04827714875398303, σ=0.9256810813636542)
```

Since `x` is a random draw from `Normal`, it's easy to check that the fitted values are sensible. Indeed, the estimates [0.04, 1.12] are close to the true values of [0.0, 1.0] that we used to generate `x`.
