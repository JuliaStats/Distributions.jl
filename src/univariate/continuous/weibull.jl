doc"""
    Weibull(α,θ)

The *Weibull distribution* with shape `α` and scale `θ` has probability density function

$f(x; \alpha, \theta) = \frac{\alpha}{\theta} \left( \frac{x}{\theta} \right)^{\alpha-1} e^{-(x/\theta)^\alpha},
    \quad x \ge 0$

```julia
Weibull()        # Weibull distribution with unit shape and unit scale, i.e. Weibull(1.0, 1.0)
Weibull(a)       # Weibull distribution with shape a and unit scale, i.e. Weibull(a, 1.0)
Weibull(a, b)    # Weibull distribution with shape a and scale b

params(d)        # Get the parameters, i.e. (a, b)
shape(d)         # Get the shape parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Weibull distribution on Wikipedia](http://en.wikipedia.org/wiki/Weibull_distribution)

"""
immutable Weibull <: ContinuousUnivariateDistribution
    α::Float64   # shape
    θ::Float64   # scale

    function Weibull(α::Real, θ::Real)
    	@check_args(Weibull, α > zero(α) && θ > zero(θ))
    	new(α, θ)
    end
    Weibull(α::Real) = Weibull(α, 1.0)
    Weibull() = new(1.0, 1.0)
end

@distr_support Weibull 0.0 Inf


#### Parameters

shape(d::Weibull) = d.α
scale(d::Weibull) = d.θ

params(d::Weibull) = (d.α, d.θ)


#### Statistics

mean(d::Weibull) = d.θ * gamma(1.0 + 1.0 / d.α)
median(d::Weibull) = d.θ * logtwo ^ (1.0 / d.α)
mode(d::Weibull) = d.α > 1.0 ? (iα = 1.0 / d.α; d.θ * (1.0 - iα) ^ iα) : 0.0

var(d::Weibull) = d.θ^2 * gamma(1.0 + 2.0 / d.α) - mean(d)^2

function skewness(d::Weibull)
    μ = mean(d)
    σ2 = var(d)
    σ = sqrt(σ2)
    r = μ / σ
    gamma(1.0 + 3.0 / d.α) * (d.θ / σ)^3 - 3.0 * r - r^3
end

function kurtosis(d::Weibull)
    α, θ = params(d)
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    r = μ / σ
    r2 = r^2
    r4 = r2^2
    (θ / σ)^4 * gamma(1.0 + 4.0 / α) - 4.0 * γ * r - 6.0 * r2 - r4 - 3.0
end

function entropy(d::Weibull)
    α, θ = params(d)
    0.5772156649015328606 * (1.0 - 1.0 / α) + log(θ / α) + 1.0
end


#### Evaluation

function pdf(d::Weibull, x::Float64)
    if x >= 0.0
        α, θ = params(d)
        z = x / θ
        (α / θ) * z^(α - 1.0) * exp(-z^α)
    else
        0.0
    end
end

function logpdf(d::Weibull, x::Float64)
    if x >= 0.0
        α, θ = params(d)
        z = x / θ
        log(α / θ) + (α - 1.0) * log(z) - z^α
    else
        -Inf
    end
end

zv(d::Weibull, x::Float64) = (x / d.θ) ^ d.α
xv(d::Weibull, z::Float64) = d.θ * z ^ (1.0 / d.α)

cdf(d::Weibull, x::Float64) = x > 0.0 ? -expm1(-zv(d, x)) : 0.0
ccdf(d::Weibull, x::Float64) = x > 0.0 ? exp(-zv(d, x)) : 1.0
logcdf(d::Weibull, x::Float64) = x > 0.0 ? log1mexp(-zv(d, x)) : -Inf
logccdf(d::Weibull, x::Float64) = x > 0.0 ? -zv(d, x) : 0.0

quantile(d::Weibull, p::Float64) = xv(d, -log1p(-p))
cquantile(d::Weibull, p::Float64) = xv(d, -log(p))
invlogcdf(d::Weibull, lp::Float64) = xv(d, -log1mexp(lp))
invlogccdf(d::Weibull, lp::Float64) = xv(d, -lp)

function gradlogpdf(d::Weibull, x::Float64)
    if insupport(Weibull, x)
        α, θ = params(d)
        (α - 1.0) / x - α * x^(α - 1.0) / (θ^α)
    else
        0.0
    end
end


#### Sampling

rand(d::Weibull) = xv(d, randexp())
