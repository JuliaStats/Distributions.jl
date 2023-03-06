"""
    JohnsonSU(ξ, λ, γ, δ)

The Johnson's ``S_U``-distribution with parameters ξ, λ, γ and δ is a transformation of the normal distribution:

```math
z = \\gamma + \\delta \\sinh^{-1}\\Big(\\frac{x-\\xi}{\\lambda}\\Big),
```

where ``z \\sim \\mathcal{N}(0,1)`` i.e. if a random variable ``X`` is defined to be

```math
X = \\lambda\\sinh\\Bigg( \\frac{Z - \\gamma}{\\delta} \\Bigg) + \\xi,
```

where ``Z \\sim \\mathcal{N}(0,1)``,  then ``X \\sim {\\rm Johnson}(\\xi, \\lambda, \\gamma, \\delta)``.

```julia
JohnsonSU()           # Equivalent to JohnsonSU(0, 1, 0, 1)
JohnsonSU(ξ, λ, γ, δ) # JohnsonSU's S_U-distrubtion with shape parameters ξ, λ, γ and δ

params(d)           # Get the parameters, i.e. (ξ, λ, γ, δ)
shape(d)            # Get the shape parameter, i.e. ξ
scale(d)            # Get the scale parameter, i.e. λ
```

External links

* [Johnson's ``S_U``-distribution on Wikipedia](http://en.wikipedia.org/wiki/Johnson%27s_SU-distribution)
"""
struct JohnsonSU{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    λ::T
    γ::T
    δ::T
    JohnsonSU{T}(ξ::T, λ::T, γ::T, δ::T) where {T<:Real} = new{T}(ξ, λ, γ, δ)
end

function JohnsonSU(ξ::T, λ::T, γ::T, δ::T; check_args::Bool=true) where {T<:Real}
    @check_args JohnsonSU (λ, λ ≥ zero(λ)) (δ, δ ≥ zero(δ))
    return JohnsonSU{T}(ξ, λ, γ, δ)
end

JohnsonSU() = JohnsonSU{Int}(0, 1, 0, 1)
JohnsonSU(ξ::Real, λ::Real, γ::Real, δ::Real; check_args::Bool=true) = JohnsonSU(promote(ξ, λ, γ, δ)...; check_args=check_args)

@distr_support JohnsonSU -Inf Inf

#### Conversions

Base.convert(::Type{JohnsonSU{T}}, d::JohnsonSU) where {T<:Real} = JohnsonSU{T}(T(d.ξ), T(d.λ), T(d.γ), T(d.δ))
Base.convert(::Type{JohnsonSU{T}}, d::JohnsonSU{T}) where {T<:Real} = d

#### Parameters

shape(d::JohnsonSU) = d.ξ
scale(d::JohnsonSU) = d.λ

params(d::JohnsonSU) = (d.ξ, d.λ, d.γ, d.δ)
partype(d::JohnsonSU{T}) where {T<:Real} = T

#### Statistics

function mean(d::JohnsonSU)
    a = exp(1/(2*d.δ^2))
    r = d.γ/d.δ
    d.ξ - d.λ * a * sinh(r)
end
function median(d::JohnsonSU)
    r = d.γ/d.δ
    d.ξ + d.λ * sinh(-r)
end
function var(d::JohnsonSU)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    d.λ^2/2 * (a-1) * (a*cosh(2r)+1)
end
function skewness(d::JohnsonSU)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    - (d.λ^3 * sqrt(a) * (a-1)^2 * (a*(a+2)*sinh(3r)+3sinh(2r))) / 4sqrt(var(d)^3)
end
function kurtosis(d::JohnsonSU)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    K1 = a^2 * (a^4+2a^3+3a^2-3) * cosh(4r)
    K2 = 4a^2 * (a+1) * cosh(3r)
    K3 = 3(2a+1)
    d.λ^3 * (a-1)^2 * (K1+K2+K3) / 8var(d)^2
end

#### Evaluation

yval(d::JohnsonSU, x::Real) = (x - d.ξ) / d.λ
zval(d::JohnsonSU, x::Real) = d.γ + d.δ * asinh(yval(d, x))
xval(d::JohnsonSU, x::Real) = d.λ * sinh((x - d.γ) / d.δ) + d.ξ

pdf(d::JohnsonSU, x::Real) = d.δ / d.λ / sqrt(1 + yval(d, x)^2) * normpdf(zval(d, x))
logpdf(d::JohnsonSU, x::Real) = log(d.δ) - log(d.λ) - 1/2log(1 + yval(d, x)^2) + normlogpdf(zval(d, x))
cdf(d::JohnsonSU, x::Real) = normcdf(zval(d, x))
logcdf(d::JohnsonSU, x::Real) = normlogcdf(zval(d, x))
ccdf(d::JohnsonSU, x::Real) = normccdf(zval(d, x))
logccdf(d::JohnsonSU, x::Real) = normlogccdf(zval(d, x))

quantile(d::JohnsonSU, q::Real) = xval(d, norminvcdf(q))
cquantile(d::JohnsonSU, p::Real) = xval(d, norminvccdf(p))
invlogcdf(d::JohnsonSU, lp::Real) = xval(d, norminvlogcdf(lp))
invlogccdf(d::JohnsonSU, lq::Real) = xval(d, norminvlogccdf(lq))

# entropy(d::JohnsonSU)
# mgf(d::JohnsonSU)
# cf(d::JohnsonSU)

#### Sampling

rand(rng::AbstractRNG, d::JohnsonSU) = xval(d, randn(rng))

## Fitting

# function fit_mle(::Type{<:JohnsonSU}, x::AbstractArray{T}) where T<:Real
