"""
    Johnson(ξ, λ, γ, δ)

The Johnson's ``S_U``-distribution with parameters ξ, λ, γ and δ is a transformation of the normal distribution:

```math
z = \\gamma + \\delta \\sinh^{-1}\\Big(\\frac{x-\\xi}{\\lambda}\\Big),
```

where ``z \\sim \\mathcal{N}(0,1)``.

```julia
Johnson()           # Equivalent to Johnson(0.0, 1.0, 0.0, 1.0)
Johnson(ξ, λ, γ, δ) # Johnson's S_U-distrubtion with shape parameters ξ, λ, γ and δ

params(d)           # Get the parameters, i.e. (ξ, λ, γ, δ)
shape(d)            # Get the shape parameter, i.e. ξ
scale(d)            # Get the scale parameter, i.e. λ
```

External links

* [Johnson's ``S_U``-distribution on Wikipedia](http://en.wikipedia.org/wiki/Johnson%27s_SU-distribution)
"""
struct Johnson{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    λ::T
    γ::T
    δ::T
    Johnson{T}(ξ::T, λ::T, γ::T, δ::T) where {T<:Real} = new{T}(ξ, λ, γ, δ)
end

function Johnson(ξ::T, λ::T, γ::T, δ::T; check_args::Bool=true) where {T<:Real}
    @check_args Johnson (λ, λ ≥ zero(λ)) (δ, δ ≥ zero(δ))
    return Johnson{T}(ξ, λ, γ, δ)
end

Johnson() = Johnson{Int}(0, 1, 0, 1)
Johnson(ξ::Real, λ::Real, γ::Real, δ::Real; check_args::Bool=true) = Johnson(promote(ξ, λ, γ, δ)...; check_args=check_args)

@distr_support Johnson -Inf Inf

#### Conversions

Base.convert(::Type{Johnson{T}}, d::Johnson) where {T<:Real} = Johnson{T}(T(d.ξ), T(d.λ), T(d.γ), T(d.δ))
Base.convert(::Type{Johnson{T}}, d::Johnson{T}) where {T<:Real} = d

#### Parameters

shape(d::Johnson) = d.ξ
scale(d::Johnson) = d.λ

params(d::Johnson) = (d.ξ, d.λ, d.γ, d.δ)
partype(d::Johnson{T}) where {T<:Real} = T

#### Statistics

function mean(d::Johnson)
    a = exp(1/(2*d.δ^2))
    r = d.γ/d.δ
    d.ξ - d.λ * a * sinh(r)
end
function median(d::Johnson)
    r = d.γ/d.δ
    d.ξ + d.λ * sinh(-r)
end
function var(d::Johnson)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    d.λ^2/2 * (a-1) * (a*cosh(2r)+1)
end
function skewness(d::Johnson)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    - (d.λ^3 * sqrt(a) * (a-1)^2 * (a*(a+2)*sinh(3r)+3sinh(2r))) / 4sqrt(var(d)^3)
end
function kurtosis(d::Johnson)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    K1 = a^2 * (a^4+2a^3+3a^2-3) * cosh(4r)
    K2 = 4a^2 * (a+1) * cosh(3r)
    K3 = 3(2a+1)
    d.λ^3 * (a-1)^2 * (K1+K2+K3) / 8var(d)^2
end

#### Evaluation

yval(d::Johnson, x::Real) = (x - d.ξ) / d.λ
zval(d::Johnson, x::Real) = d.γ + d.δ * asinh(yval(d, x))
xval(d::Johnson, x::Real) = d.λ * sinh((x - d.γ) / d.δ) + d.ξ

pdf(d::Johnson, x::Real) = d.δ / d.λ / sqrt(1 + yval(d, x)^2) * normpdf(zval(d, x))
logpdf(d::Johnson, x::Real) = log(d.δ) - log(d.λ) - 1/2log(1 + yval(d, x)^2) + normlogpdf(zval(d, x))
cdf(d::Johnson, x::Real) = normcdf(zval(d, x))
logcdf(d::Johnson, x::Real) = normlogcdf(zval(d, x))
ccdf(d::Johnson, x::Real) = normccdf(zval(d, x))
logccdf(d::Johnson, x::Real) = normlogccdf(zval(d, x))

quantile(d::Johnson, q::Real) = xval(d, norminvcdf(q))
cquantile(d::Johnson, p::Real) = xval(d, norminvccdf(p))
invlogcdf(d::Johnson, lp::Real) = xval(d, norminvlogcdf(lp))
invlogccdf(d::Johnson, lq::Real) = xval(d, norminvlogccdf(lq))

# entropy(d::Johnson)
# mgf(d::Johnson)
# cf(d::Johnson)

#### Sampling

rand(rng::AbstractRNG, d::Johnson) = xval(d, randn(rng))

## Fitting

# function fit_mle(::Type{<:Johnson}, x::AbstractArray{T}) where T<:Real
