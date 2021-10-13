"""
    SymTriangularDist(μ, σ)

The *Symmetric triangular distribution* with location `μ` and scale `σ` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sigma} \\left( 1 - \\left| \\frac{x - \\mu}{\\sigma} \\right| \\right), \\quad \\mu - \\sigma \\le x \\le \\mu + \\sigma
```

```julia
SymTriangularDist()         # Symmetric triangular distribution with zero location and unit scale
SymTriangularDist(μ)        # Symmetric triangular distribution with location μ and unit scale
SymTriangularDist(μ, s)     # Symmetric triangular distribution with location μ and scale σ

params(d)       # Get the parameters, i.e. (μ, σ)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
```
"""
struct SymTriangularDist{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    SymTriangularDist{T}(µ::T, σ::T) where {T <: Real} = new{T}(µ, σ)
end

function SymTriangularDist(μ::T, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(SymTriangularDist, σ > zero(σ))
    return SymTriangularDist{T}(μ, σ)
end

SymTriangularDist(μ::Real, σ::Real) = SymTriangularDist(promote(μ, σ)...)
SymTriangularDist(μ::Integer, σ::Integer) = SymTriangularDist(float(μ), float(σ))
SymTriangularDist(μ::T) where {T <: Real} = SymTriangularDist(μ, one(T))
SymTriangularDist() = SymTriangularDist(0.0, 1.0, check_args=false)

@distr_support SymTriangularDist d.μ - d.σ d.μ + d.σ

#### Conversions

function convert(::Type{SymTriangularDist{T}}, μ::Real, σ::Real) where T<:Real
    SymTriangularDist(T(μ), T(σ))
end
function convert(::Type{SymTriangularDist{T}}, d::SymTriangularDist{S}) where {T <: Real, S <: Real}
    SymTriangularDist(T(d.μ), T(d.σ), check_args=false)
end

#### Parameters

location(d::SymTriangularDist) = d.μ
scale(d::SymTriangularDist) = d.σ

params(d::SymTriangularDist) = (d.μ, d.σ)
@inline partype(d::SymTriangularDist{T}) where {T<:Real} = T


#### Statistics

mean(d::SymTriangularDist) = d.μ
median(d::SymTriangularDist) = d.μ
mode(d::SymTriangularDist) = d.μ

var(d::SymTriangularDist) = d.σ^2 / 6
skewness(d::SymTriangularDist{T}) where {T<:Real} = zero(T)
kurtosis(d::SymTriangularDist{T}) where {T<:Real} = T(-3)/5

entropy(d::SymTriangularDist) = 1//2 + log(d.σ)


#### Evaluation

zval(d::SymTriangularDist, x::Real) = min(abs(x - d.μ) / d.σ, 1)
xval(d::SymTriangularDist, z::Real) = d.μ + z * d.σ

pdf(d::SymTriangularDist, x::Real) = (1 - zval(d, x)) / scale(d)
logpdf(d::SymTriangularDist, x::Real) = log(pdf(d, x))

function cdf(d::SymTriangularDist, x::Real)
    r = (1 - zval(d, x))^2/2
    return x < d.μ ? r : 1 - r
end

function ccdf(d::SymTriangularDist, x::Real)
    r = (1 - zval(d, x))^2/2
    return x < d.μ ? 1 - r : r
end

function logcdf(d::SymTriangularDist, x::Real)
    log_r = 2 * log1p(- zval(d, x)) + loghalf
    return x < d.μ ? log_r : log1mexp(log_r)
end

function logccdf(d::SymTriangularDist, x::Real)
    log_r = 2 * log1p(- zval(d, x)) + loghalf
    return x < d.μ ? log1mexp(log_r) : log_r
end

quantile(d::SymTriangularDist, p::Real) = p < 1/2 ? xval(d, sqrt(2p) - 1) :
                                                       xval(d, 1 - sqrt(2(1 - p)))

cquantile(d::SymTriangularDist, p::Real) = p > 1/2 ? xval(d, sqrt(2(1-p)) - 1) :
                                                        xval(d, 1 - sqrt(2p))

invlogcdf(d::SymTriangularDist, lp::Real) = lp < loghalf ? xval(d, expm1(1/2*(lp - loghalf))) :
                                                              xval(d, 1 - sqrt(-2expm1(lp)))

function invlogccdf(d::SymTriangularDist, lp::Real)
    lp > loghalf ? xval(d, sqrt(-2*expm1(lp)) - 1) :
    xval(d, -(expm1((lp - loghalf)/2)))
end


function mgf(d::SymTriangularDist, t::Real)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return one(a)
    4*exp(μ * t) * (sinh(a/2) / a)^2
end

function cf(d::SymTriangularDist, t::Real)
    (μ, σ) = params(d)
    a = σ * t
    a == zero(a) && return complex(one(a))
    4*cis(μ * t) * (sin(a/2) / a)^2
end


#### Sampling

rand(rng::AbstractRNG, d::SymTriangularDist) = xval(d, rand(rng) - rand(rng))
