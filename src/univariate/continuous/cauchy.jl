doc"""
    Cauchy(μ, σ)

The *Cauchy distribution* with location `μ` and scale `σ` has probability density function

$f(x; \mu, \sigma) = \frac{1}{\pi \sigma \left(1 + \left(\frac{x - \mu}{\sigma} \right)^2 \right)}$

```julia
Cauchy()         # Standard Cauchy distribution, i.e. Cauchy(0, 1)
Cauchy(u)        # Cauchy distribution with location u and unit scale, i.e. Cauchy(u, 1)
Cauchy(u, b)     # Cauchy distribution with location u and scale b

params(d)        # Get the parameters, i.e. (u, b)
location(d)      # Get the location parameter, i.e. u
scale(d)         # Get the scale parameter, i.e. b
```

External links

* [Cauchy distribution on Wikipedia](http://en.wikipedia.org/wiki/Cauchy_distribution)

"""

immutable Cauchy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    function (::Type{Cauchy{T}}){T}(μ::T, σ::T)
        @check_args(Cauchy, σ > zero(σ))
        new{T}(μ, σ)
    end
end

Cauchy{T<:Real}(μ::T, σ::T) = Cauchy{T}(μ, σ)
Cauchy(μ::Real, σ::Real) = Cauchy(promote(μ, σ)...)
Cauchy(μ::Integer, σ::Integer) = Cauchy(Float64(μ), Float64(σ))
Cauchy(μ::Real) = Cauchy(μ, 1.0)
Cauchy() = Cauchy(0.0, 1.0)

@distr_support Cauchy -Inf Inf

#### Conversions
function convert{T<:Real}(::Type{Cauchy{T}}, μ::Real, σ::Real)
    Cauchy(T(μ), T(σ))
end
function convert{T <: Real, S <: Real}(::Type{Cauchy{T}}, d::Cauchy{S})
    Cauchy(T(d.μ), T(d.σ))
end

#### Parameters

location(d::Cauchy) = d.μ
scale(d::Cauchy) = d.σ

params(d::Cauchy) = (d.μ, d.σ)
@inline partype{T<:Real}(d::Cauchy{T}) = T


#### Statistics

mean{T<:Real}(d::Cauchy{T}) = T(NaN)
median(d::Cauchy) = d.μ
mode(d::Cauchy) = d.μ

var{T<:Real}(d::Cauchy{T}) = T(NaN)
skewness{T<:Real}(d::Cauchy{T}) = T(NaN)
kurtosis{T<:Real}(d::Cauchy{T}) = T(NaN)

entropy(d::Cauchy) = log4π + log(d.σ)


#### Functions

zval(d::Cauchy, x::Real) = (x - d.μ) / d.σ
xval(d::Cauchy, z::Real) = d.μ + z * d.σ

pdf(d::Cauchy, x::Real) = 1 / (π * scale(d) * (1 + zval(d, x)^2))
logpdf(d::Cauchy, x::Real) = - (log1psq(zval(d, x)) + logπ + log(d.σ))

function cdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan2(x - μ, σ) + 1//2
end

function ccdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan2(μ - x, σ) + 1//2
end

function quantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (p - 1//2))
end

function cquantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (1//2 - p))
end

mgf{T<:Real}(d::Cauchy{T}, t::Real) = t == zero(t) ? one(T) : T(NaN)
cf(d::Cauchy, t::Real) = exp(im * (t * d.μ) - d.σ * abs(t))


#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit{T<:Real}(::Type{Cauchy}, x::AbstractArray{T})
    l, m, u = quantile(x, [0.25, 0.5, 0.75])
    Cauchy(m, (u - l) / 2)
end
