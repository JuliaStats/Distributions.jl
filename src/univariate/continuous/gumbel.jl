doc"""
    Gumbel(μ, θ)

The *Gumbel distribution*  with location `μ` and scale `θ` has probability density function

$f(x; \mu, \theta) = \frac{1}{\theta} e^{-(z + e^z)},
\quad \text{ with } z = \frac{x - \mu}{\theta}$

```julia
Gumbel()            # Gumbel distribution with zero location and unit scale, i.e. Gumbel(0, 1)
Gumbel(u)           # Gumbel distribution with location u and unit scale, i.e. Gumbel(u, 1)
Gumbel(u, b)        # Gumbel distribution with location u and scale b

params(d)        # Get the parameters, i.e. (u, b)
location(d)      # Get the location parameter, i.e. u
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Gumbel distribution on Wikipedia](http://en.wikipedia.org/wiki/Gumbel_distribution)
"""
immutable Gumbel{T<:Real} <: ContinuousUnivariateDistribution
    μ::T  # location
    θ::T  # scale

    Gumbel(μ::T, θ::T) = (@check_args(Gumbel, θ > zero(θ)); new(μ, θ))
end

Gumbel{T<:Real}(μ::T, θ::T) = Gumbel{T}(μ, θ)
Gumbel(μ::Real, θ::Real) = Gumbel(promote(μ, θ)...)
Gumbel(μ::Integer, θ::Integer) = Gumbel(Float64(μ), Float64(θ))
Gumbel(μ::Real) = Gumbel(μ, 1.0)
Gumbel() = Gumbel(0.0, 1.0)

@distr_support Gumbel -Inf Inf

const DoubleExponential = Gumbel

#### Conversions

convert{T <: Real, S <: Real}(::Type{Gumbel{T}}, μ::S, θ::S) = Gumbel(T(μ), T(θ))
convert{T <: Real, S <: Real}(::Type{Gumbel{T}}, d::Gumbel{S}) = Gumbel(T(d.μ), T(d.θ))

#### Parameters

location(d::Gumbel) = d.μ
scale(d::Gumbel) = d.θ
params(d::Gumbel) = (d.μ, d.θ)
@inline partype{T<:Real}(d::Gumbel{T}) = T


#### Statistics

mean(d::Gumbel) = d.μ + d.θ * γ

median{T<:Real}(d::Gumbel{T}) = d.μ - d.θ * log(T(logtwo))

mode(d::Gumbel) = d.μ

var{T<:Real}(d::Gumbel{T}) = T(π)^2/6 * d.θ^2

skewness{T<:Real}(d::Gumbel{T}) = 112*sqrt(T(6))*zeta(T(3))/π/π/π

kurtosis{T<:Real}(d::Gumbel{T}) = T(12)/5

entropy(d::Gumbel) = log(d.θ) + 1 + γ


#### Evaluation

zval(d::Gumbel, x::Real) = (x - d.μ) / d.θ
xval(d::Gumbel, z::Real) = x * d.θ + d.μ

function pdf(d::Gumbel, x::Real)
    z = zval(d, x)
    exp(-z - exp(-z)) / d.θ
end

function logpdf(d::Gumbel, x::Real)
    z = zval(d, x)
    - (z + exp(-z) + log(d.θ))
end

cdf(d::Gumbel, x::Real) = exp(-exp(-zval(d, x)))
logcdf(d::Gumbel, x::Real) = -exp(-zval(d, x))

quantile(d::Gumbel, p::Real) = d.μ - d.θ * log(-log(p))

gradlogpdf(d::Gumbel, x::Real) = - (1 + exp((d.μ - x) / d.θ)) / d.θ


#### Sampling

rand(d::Gumbel) = quantile(d, rand())
