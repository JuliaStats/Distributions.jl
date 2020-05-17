"""
    Weibull(α,θ)

The *Weibull distribution* with shape `α` and scale `θ` has probability density function

```math
f(x; \\alpha, \\theta) = \\frac{\\alpha}{\\theta} \\left( \\frac{x}{\\theta} \\right)^{\\alpha-1} e^{-(x/\\theta)^\\alpha},
    \\quad x \\ge 0
```

```julia
Weibull()        # Weibull distribution with unit shape and unit scale, i.e. Weibull(1, 1)
Weibull(a)       # Weibull distribution with shape a and unit scale, i.e. Weibull(a, 1)
Weibull(a, b)    # Weibull distribution with shape a and scale b

params(d)        # Get the parameters, i.e. (a, b)
shape(d)         # Get the shape parameter, i.e. a
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Weibull distribution on Wikipedia](http://en.wikipedia.org/wiki/Weibull_distribution)

"""
struct Weibull{T<:Real} <: ContinuousUnivariateDistribution
    α::T   # shape
    θ::T   # scale

    function Weibull{T}(α::T, θ::T) where {T <: Real}
        new{T}(α, θ)
    end
end

function Weibull(α::T, θ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Weibull, α > zero(α) && θ > zero(θ))
    return Weibull{T}(α, θ)
end

Weibull(α::Real, θ::Real) = Weibull(promote(α, θ)...)
Weibull(α::Integer, θ::Integer) = Weibull(float(α), float(θ))
Weibull(α::T) where {T <: Real} = Weibull(α, one(T))
Weibull() = Weibull(1.0, 1.0, check_args=false)

@distr_support Weibull 0.0 Inf

#### Conversions

convert(::Type{Weibull{T}}, α::Real, θ::Real) where {T<:Real} = Weibull(T(α), T(θ))
convert(::Type{Weibull{T}}, d::Weibull{S}) where {T <: Real, S <: Real} = Weibull(T(d.α), T(d.θ), check_args=false)

#### Parameters

shape(d::Weibull) = d.α
scale(d::Weibull) = d.θ

params(d::Weibull) = (d.α, d.θ)
partype(::Weibull{T}) where {T<:Real} = T


#### Statistics

mean(d::Weibull) = d.θ * gamma(1 + 1/d.α)
median(d::Weibull) = d.θ * logtwo ^ (1/d.α)
mode(d::Weibull{T}) where {T<:Real} = d.α > 1 ? (iα = 1 / d.α; d.θ * (1 - iα)^iα) : zero(T)

var(d::Weibull) = d.θ^2 * gamma(1 + 2/d.α) - mean(d)^2

function skewness(d::Weibull)
    μ = mean(d)
    σ2 = var(d)
    σ = sqrt(σ2)
    r = μ / σ
    gamma(1 + 3/d.α) * (d.θ/σ)^3 - 3r - r^3
end

function kurtosis(d::Weibull)
    α, θ = params(d)
    μ = mean(d)
    σ = std(d)
    γ = skewness(d)
    r = μ / σ
    r2 = r^2
    r4 = r2^2
    (θ/σ)^4 * gamma(1 + 4/α) - 4γ*r - 6r2 - r4 - 3
end

function entropy(d::Weibull)
    α, θ = params(d)
    0.5772156649015328606 * (1 - 1/α) + log(θ/α) + 1
end


#### Evaluation

function pdf(d::Weibull{T}, x::Real) where T<:Real
    if x >= 0
        α, θ = params(d)
        z = x / θ
        (α / θ) * z^(α - 1) * exp(-z^α)
    else
        zero(T)
    end
end

function logpdf(d::Weibull{T}, x::Real) where T<:Real
    if x >= 0
        α, θ = params(d)
        z = x / θ
        log(α / θ) + (α - 1) * log(z) - z^α
    else
        -T(Inf)
    end
end

zv(d::Weibull, x::Real) = (x / d.θ) ^ d.α
xv(d::Weibull, z::Real) = d.θ * z ^ (1 / d.α)

cdf(d::Weibull{T}, x::Real) where {T<:Real} = x > 0 ? -expm1(-zv(d, x)) : zero(T)
ccdf(d::Weibull{T}, x::Real) where {T<:Real} = x > 0 ? exp(-zv(d, x)) : one(T)
logcdf(d::Weibull{T}, x::Real) where {T<:Real} = x > 0 ? log1mexp(-zv(d, x)) : -T(Inf)
logccdf(d::Weibull{T}, x::Real) where {T<:Real} = x > 0 ? -zv(d, x) : zero(T)

quantile(d::Weibull, p::Real) = xv(d, -log1p(-p))
cquantile(d::Weibull, p::Real) = xv(d, -log(p))
invlogcdf(d::Weibull, lp::Real) = xv(d, -log1mexp(lp))
invlogccdf(d::Weibull, lp::Real) = xv(d, -lp)

function gradlogpdf(d::Weibull{T}, x::Real) where T<:Real
    if insupport(Weibull, x)
        α, θ = params(d)
        (α - 1) / x - α * x^(α - 1) / (θ^α)
    else
        zero(T)
    end
end


#### Sampling

rand(rng::AbstractRNG, d::Weibull) = xv(d, randexp(rng))
