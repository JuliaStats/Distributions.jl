doc"""
    Skellam(μ1, μ2)

A *Skellam distribution* describes the difference between two independent [`Poisson`](:func:`Poisson`) variables, respectively with rate `μ1` and `μ2`.

$P(X = k) = e^{-(\mu_1 + \mu_2)} \left( \frac{\mu_1}{\mu_2} \right)^{k/2} I_k(2 \sqrt{\mu_1 \mu_2}) \quad \text{for integer } k$

where $I_k$ is the modified Bessel function of the first kind.

```julia
Skellam(mu1, mu2)   # Skellam distribution for the difference between two Poisson variables,
                    # respectively with expected values mu1 and mu2.

params(d)           # Get the parameters, i.e. (mu1, mu2)
```

External links:

* [Skellam distribution on Wikipedia](http://en.wikipedia.org/wiki/Skellam_distribution)
"""
immutable Skellam <: DiscreteUnivariateDistribution
    μ1::Float64
    μ2::Float64

    function Skellam(μ1::Real, μ2::Real)
        @check_args(Skellam, μ1 > zero(μ1) && μ2 > zero(μ2))
        new(μ1, μ2)
    end
    Skellam(μ::Real) = Skellam(μ, μ)
    Skellam() = new(1.0, 1.0)
end

@distr_support Skellam -Inf Inf

### Parameters

params(d::Skellam) = (d.μ1, d.μ2)


### Statistics

mean(d::Skellam) = d.μ1 - d.μ2

var(d::Skellam) = d.μ1 + d.μ2

skewness(d::Skellam) = mean(d) / (var(d)^1.5)

kurtosis(d::Skellam) = 1.0 / var(d)


### Evaluation

function logpdf(d::Skellam, x::Int)
    μ1, μ2 = params(d)
    - (μ1 + μ2) + (x / 2.0) * log(μ1 / μ2) + log(besseli(x, 2.0 * sqrt(μ1) * sqrt(μ2)))
end

pdf(d::Skellam, x::Int) = exp(logpdf(d, x))

function mgf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(-(μ1 + μ2) + μ1 * exp(t) + μ2 * exp(-t))
end

function cf(d::Skellam, t::Real)
    μ1, μ2 = params(d)
    exp(-(μ1 + μ2) + μ1 * cis(t) + μ2 * cis(-t))
end

### Sampling

rand(d::Skellam) = rand(Poisson(d.μ1)) - rand(Poisson(d.μ2))
