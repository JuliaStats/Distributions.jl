"""
    Cauchy(μ, σ)

The *Cauchy distribution* with location `μ` and scale `σ` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\pi \\sigma \\left(1 + \\left(\\frac{x - \\mu}{\\sigma} \\right)^2 \\right)}
```

```julia
Cauchy()         # Standard Cauchy distribution, i.e. Cauchy(0, 1)
Cauchy(μ)        # Cauchy distribution with location μ and unit scale, i.e. Cauchy(μ, 1)
Cauchy(μ, σ)     # Cauchy distribution with location μ and scale σ

params(d)        # Get the parameters, i.e. (μ, σ)
location(d)      # Get the location parameter, i.e. μ
scale(d)         # Get the scale parameter, i.e. σ
```

External links

* [Cauchy distribution on Wikipedia](http://en.wikipedia.org/wiki/Cauchy_distribution)

"""
struct Cauchy{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Cauchy{T}(µ, σ) where {T} = new{T}(µ, σ)
end

function Cauchy(μ::T, σ::T; check_args::Bool=true) where {T<:Real}
    @check_args Cauchy (σ, σ > zero(σ))
    return Cauchy{T}(μ, σ)
end

Cauchy(μ::Real, σ::Real; check_args::Bool=true) = Cauchy(promote(μ, σ)...; check_args=check_args)
Cauchy(μ::Integer, σ::Integer; check_args::Bool=true) = Cauchy(float(μ), float(σ); check_args=check_args)
Cauchy(μ::Real=0.0) = Cauchy(μ, one(μ); check_args=false)

@distr_support Cauchy -Inf Inf

#### Conversions
function convert(::Type{Cauchy{T}}, μ::Real, σ::Real) where T<:Real
    Cauchy(T(μ), T(σ))
end
Base.convert(::Type{Cauchy{T}}, d::Cauchy) where {T<:Real} = Cauchy{T}(T(d.μ), T(d.σ))
Base.convert(::Type{Cauchy{T}}, d::Cauchy{T}) where {T<:Real} = d

#### Parameters

location(d::Cauchy) = d.μ
scale(d::Cauchy) = d.σ

params(d::Cauchy) = (d.μ, d.σ)
@inline partype(d::Cauchy{T}) where {T<:Real} = T


#### Statistics

mean(d::Cauchy{T}) where {T<:Real} = T(NaN)
median(d::Cauchy) = d.μ
mode(d::Cauchy) = d.μ

var(d::Cauchy{T}) where {T<:Real} = T(NaN)
skewness(d::Cauchy{T}) where {T<:Real} = T(NaN)
kurtosis(d::Cauchy{T}) where {T<:Real} = T(NaN)

entropy(d::Cauchy) = log4π + log(d.σ)


#### Functions

zval(d::Cauchy, x::Real) = (x - d.μ) / d.σ
xval(d::Cauchy, z::Real) = d.μ + z * d.σ

pdf(d::Cauchy, x::Real) = 1 / (π * scale(d) * (1 + zval(d, x)^2))
logpdf(d::Cauchy, x::Real) = - (log1psq(zval(d, x)) + logπ + log(d.σ))

function cdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan(x - μ, σ) + 1//2
end

function ccdf(d::Cauchy, x::Real)
    μ, σ = params(d)
    invπ * atan(μ - x, σ) + 1//2
end

function quantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (p - 1//2))
end

function cquantile(d::Cauchy, p::Real)
    μ, σ = params(d)
    μ + σ * tan(π * (1//2 - p))
end

mgf(d::Cauchy{T}, t::Real) where {T<:Real} = t == zero(t) ? one(T) : T(NaN)
cf(d::Cauchy, t::Real) = exp(im * (t * d.μ) - d.σ * abs(t))

#### Affine transformations

Base.:+(d::Cauchy, c::Real) = Cauchy(d.μ + c, d.σ)
Base.:*(c::Real, d::Cauchy) = Cauchy(c * d.μ, abs(c) * d.σ)

#### Fitting

# Note: this is not a Maximum Likelihood estimator
function fit(::Type{<:Cauchy}, x::AbstractArray{T}) where T<:Real
    l, m, u = quantile(x, [0.25, 0.5, 0.75])
    Cauchy(m, (u - l) / 2)
end
