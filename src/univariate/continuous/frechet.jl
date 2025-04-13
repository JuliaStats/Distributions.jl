"""
    Frechet(α,θ)

The *Fréchet distribution* with shape `α` and scale `θ` has probability density function

```math
f(x; \\alpha, \\theta) = \\frac{\\alpha}{\\theta} \\left( \\frac{x}{\\theta} \\right)^{-\\alpha-1}
e^{-(x/\\theta)^{-\\alpha}}, \\quad x > 0
```

```julia
Frechet()        # Fréchet distribution with unit shape and unit scale, i.e. Frechet(1, 1)
Frechet(α)       # Fréchet distribution with shape α and unit scale, i.e. Frechet(α, 1)
Frechet(α, θ)    # Fréchet distribution with shape α and scale θ

params(d)        # Get the parameters, i.e. (α, θ)
shape(d)         # Get the shape parameter, i.e. α
scale(d)         # Get the scale parameter, i.e. θ
```

External links

* [Fréchet_distribution on Wikipedia](http://en.wikipedia.org/wiki/Fréchet_distribution)

"""
struct Frechet{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    θ::T
    Frechet{T}(α::T, θ::T) where {T<:Real} = new{T}(α, θ)
end

function Frechet(α::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Frechet (α, α > zero(α)) (θ, θ > zero(θ))
    return Frechet{T}(α, θ)
end

Frechet(α::Real, θ::Real; check_args::Bool=true) = Frechet(promote(α, θ)...; check_args=check_args)
Frechet(α::Integer, θ::Integer; check_args::Bool=true) = Frechet(float(α), float(θ); check_args=check_args)
Frechet(α::Real=1.0) = Frechet(α, one(α); check_args=false)

@distr_support Frechet 0.0 Inf

#### Conversions
function convert(::Type{Frechet{T}}, α::S, θ::S) where {T <: Real, S <: Real}
    Frechet(T(α), T(θ))
end
Base.convert(::Type{Frechet{T}}, d::Frechet) where {T<:Real} = Frechet{T}(T(d.α), T(d.θ))
Base.convert(::Type{Frechet{T}}, d::Frechet{T}) where {T<:Real} = d

#### Parameters

shape(d::Frechet) = d.α
scale(d::Frechet) = d.θ
params(d::Frechet) = (d.α, d.θ)
partype(::Frechet{T}) where {T} = T


#### Statistics

function mean(d::Frechet{T}) where {T}
    α = d.α
    return α > 1 ? d.θ * gamma(1 - 1 / α) : T(Inf)
end

median(d::Frechet) = d.θ * logtwo^(-1 / d.α)

mode(d::Frechet) = (iα = -1/d.α; d.θ * (1 - iα) ^ iα)

function var(d::Frechet{T}) where {T<:Real}
    if d.α > 2
        iα = 1 / d.α
        return d.θ^2 * (gamma(1 - 2 * iα) - gamma(1 - iα)^2)
    else
        return T(Inf)
    end
end

function skewness(d::Frechet{T}) where T<:Real
    if d.α > 3
        iα = 1 / d.α
        g1 = gamma(1 - iα)
        g2 = gamma(1 - 2 * iα)
        g3 = gamma(1 - 3 * iα)
        return (g3 - 3g2 * g1 + 2 * g1^3) / ((g2 - g1^2)^1.5)
    else
        return T(Inf)
    end
end

function kurtosis(d::Frechet{T}) where T<:Real
    if d.α > 3
        iα = 1 / d.α
        g1 = gamma(1 - iα)
        g2 = gamma(1 - 2iα)
        g3 = gamma(1 - 3iα)
        g4 = gamma(1 - 4iα)
        return (g4 - 4g3 * g1 + 3 * g2^2) / ((g2 - g1^2)^2) - 6
    else
        return T(Inf)
    end
end

function entropy(d::Frechet)
    1 + MathConstants.γ / d.α + MathConstants.γ + log(d.θ / d.α)
end


#### Evaluation

function logpdf(d::Frechet{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    if x > 0
        z = θ / x
        return log(α / θ) + (1 + α) * log(z) - z^α
    else
        return -T(Inf)
    end
end

zval(d::Frechet, x::Real) = (d.θ / max(x, 0))^d.α
xval(d::Frechet, z::Real) = d.θ * z^(- 1 / d.α)

cdf(d::Frechet, x::Real) = exp(- zval(d, x))
ccdf(d::Frechet, x::Real) = -expm1(- zval(d, x))
logcdf(d::Frechet, x::Real) = - zval(d, x)
logccdf(d::Frechet, x::Real) = log1mexp(- zval(d, x))

quantile(d::Frechet, p::Real) = xval(d, -log(p))
cquantile(d::Frechet, p::Real) = xval(d, -log1p(-p))
invlogcdf(d::Frechet, lp::Real) = xval(d, -lp)
invlogccdf(d::Frechet, lp::Real) = xval(d, -log1mexp(lp))

function gradlogpdf(d::Frechet{T}, x::Real) where T<:Real
    (α, θ) = params(d)
    insupport(Frechet, x) ? -(α + 1) / x + α * (θ^α) * x^(-α-1)  : zero(T)
end

## Sampling

rand(rng::AbstractRNG, d::Frechet) = d.θ * randexp(rng) ^ (-1 / d.α)
