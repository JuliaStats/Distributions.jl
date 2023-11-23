"""
    Logistic(μ,θ)

The *Logistic distribution* with location `μ` and scale `θ` has probability density function

```math
f(x; \\mu, \\theta) = \\frac{1}{4 \\theta} \\mathrm{sech}^2
\\left( \\frac{x - \\mu}{2 \\theta} \\right)
```

```julia
Logistic()       # Logistic distribution with zero location and unit scale, i.e. Logistic(0, 1)
Logistic(μ)      # Logistic distribution with location μ and unit scale, i.e. Logistic(μ, 1)
Logistic(μ, θ)   # Logistic distribution with location μ and scale θ

params(d)       # Get the parameters, i.e. (μ, θ)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. θ
```

External links

* [Logistic distribution on Wikipedia](http://en.wikipedia.org/wiki/Logistic_distribution)

"""
struct Logistic{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    θ::T
    Logistic{T}(µ::T, θ::T) where {T} = new{T}(µ, θ)
end


function Logistic(μ::T, θ::T; check_args::Bool=true) where {T <: Real}
    @check_args Logistic (θ, θ > zero(θ))
    return Logistic{T}(μ, θ)
end

Logistic(μ::Real, θ::Real; check_args::Bool=true) = Logistic(promote(μ, θ)...; check_args=check_args)
Logistic(μ::Integer, θ::Integer; check_args::Bool=true) = Logistic(float(μ), float(θ); check_args=check_args)
Logistic(μ::Real=0.0) = Logistic(μ, one(μ); check_args=false)

@distr_support Logistic -Inf Inf

#### Conversions
function convert(::Type{Logistic{T}}, μ::S, θ::S) where {T <: Real, S <: Real}
    Logistic(T(μ), T(θ))
end
function Base.convert(::Type{Logistic{T}}, d::Logistic) where {T<:Real}
    Logistic{T}(T(d.μ), T(d.θ))
end
Base.convert(::Type{Logistic{T}}, d::Logistic{T}) where {T<:Real} = d

#### Parameters

location(d::Logistic) = d.μ
scale(d::Logistic) = d.θ

params(d::Logistic) = (d.μ, d.θ)
@inline partype(d::Logistic{T}) where {T<:Real} = T


#### Statistics

mean(d::Logistic) = d.μ
median(d::Logistic) = d.μ
mode(d::Logistic) = d.μ

std(d::Logistic) = π * d.θ / sqrt3
var(d::Logistic) = (π * d.θ)^2 / 3
skewness(d::Logistic{T}) where {T<:Real} = zero(T)
kurtosis(d::Logistic{T}) where {T<:Real} = T(6)/5

entropy(d::Logistic) = log(d.θ) + 2


#### Evaluation

zval(d::Logistic, x::Real) = (x - d.μ) / d.θ
xval(d::Logistic, z::Real) = d.μ + z * d.θ

pdf(d::Logistic, x::Real) = (lz = logistic(-abs(zval(d, x))); lz*(1-lz)/d.θ)
logpdf(d::Logistic, x::Real) = (u = -abs(zval(d, x)); u - 2*log1pexp(u) - log(d.θ))

cdf(d::Logistic, x::Real) = logistic(zval(d, x))
ccdf(d::Logistic, x::Real) = logistic(-zval(d, x))
logcdf(d::Logistic, x::Real) = -log1pexp(-zval(d, x))
logccdf(d::Logistic, x::Real) = -log1pexp(zval(d, x))

quantile(d::Logistic, p::Real) = xval(d, logit(p))
cquantile(d::Logistic, p::Real) = xval(d, -logit(p))
invlogcdf(d::Logistic, lp::Real) = xval(d, -logexpm1(-lp))
invlogccdf(d::Logistic, lp::Real) = xval(d, logexpm1(-lp))

function gradlogpdf(d::Logistic, x::Real)
    e = exp(-zval(d, x))
    ((2e) / (1 + e) - 1) / d.θ
end

mgf(d::Logistic, t::Real) = exp(t * d.μ) / sinc(d.θ * t)
function cgf(d::Logistic, t)
    μ, θ = params(d)
    t*μ - log(sinc(θ*t))
end
function cf(d::Logistic, t::Real)
    a = (π * t) * d.θ
    a == zero(a) ? complex(one(a)) : cis(t * d.μ) * (a / sinh(a))
end
