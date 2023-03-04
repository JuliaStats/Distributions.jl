"""
    Johnson(ξ, λ, γ, δ)

The Johnson's ``S_U``-distribution with parameters ξ, λ, γ and δ is a transformation of the normal distribution:

```math
z \\sim \\gamma + \\delta \\sinh^{-1}\\Big(\\frac{x-\\xi}{\\lambda}\\Big),
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
	Johnson{T}(ξ, λ, γ, δ) where {T} = new{T}(ξ, λ, γ, δ)
end

function Johnson(ξ::T, λ::T, γ::T, δ::T; check_args::Bool=true) where {T<:Real}
    @check_args Johnson (λ, λ ≥ zero(λ)) (δ, δ ≥ zero(δ))
	return Johnson{T}(ξ, λ, γ, δ)
end

Johnson() = Johnson{Float64}(0.0, 1.0, 0.0, 1.0)
Johnson(ξ::Real, λ::Real, γ::Real, δ::Real; check_args::Bool=true) = Johnson(promote(ξ, λ, γ, δ)...; check_args=check_args)
Johnson(ξ::Integer, λ::Integer, γ::Integer, δ::Integer; check_args::Bool=true) = Johnson(float(ξ), float(λ), float(γ), float(δ); check_args=check_args)

@distr_support Johnson -Inf Inf

#### Conversions

convert(::Type{Johnson{T}}, ξ::S, λ::S, γ::S, δ::S) where {T<:Real, S<:Real} = Johnson(T(ξ), T(λ), T(γ), T(δ))
Base.convert(::Type{Johnson{T}}, d::Johnson) where {T<:Real} = Johnson{T}(T(d.ξ), T(d.λ), T(d.γ), T(d.δ))
Base.convert(::Type{Johnson{T}}, d::Johnson{T}) where {T<:Real} = d

#### Parameters

shape(d::Johnson) = d.ξ
scale(d::Johnson) = d.λ

params(d::Johnson) = (d.ξ, d.λ, d.γ, d.δ)
partype(d::Johnson{T}) where {T<:Real} = T

#### Statistics

function mean(d::Johnson)
    a = exp(d.δ^-2)
    r = d.γ/d.δ
    d.ξ - d.λ * sqrt(a) * sinh(r)
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
    - (d.λ^3 * sqrt(a) * (a-1)^2 * (a*(a+2)*sinh(3r)+3sinh(2r))) / 4var(d)^1.5
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

function pdf(d::Johnson, x::Real)
	y = (x - d.ξ) / d.λ
	z = d.γ + d.δ * asinh(y)
	d.δ / d.λ / sqrt(2π * (1.0 + y^2)) * exp(-0.5 * z ^ 2)
end

function logpdf(d::Johnson, x::Real)
    y = (x - d.ξ) / d.λ
    z = d.γ + d.δ * asinh(y)
    -0.5 * (z^2 + log2π + log(1.0 + y^2)) + log(d.δ) - log(d.λ)
end

function cdf(d::Johnson, x::Real)
	y = (x - d.ξ) / d.λ
	z = d.γ + d.δ * asinh(y)
	cdf(Normal(), z)
end

function ccdf(d::Johnson, x::Real)
	y = (x - d.ξ) / d.λ
	z = d.γ + d.δ * asinh(y)
	ccdf(Normal(), z)
end

function logcdf(d::Johnson, x::Real)
	y = (x - d.ξ) / d.λ
	z = d.γ + d.δ * asinh(y)
	logcdf(Normal(), z)
end

function logccdf(d::Johnson, x::Real)
	y = (x - d.ξ) / d.λ
	z = d.γ + d.δ * asinh(y)
	logccdf(Normal(), z)
end

quantile(d::Johnson, q::Real) = d.λ * sinh((quantile(Normal(), q) - d.γ) / d.δ) + d.ξ
cquantile(d::Johnson, q::Real) = d.λ * sinh((cquantile(Normal(), q) - d.γ) / d.δ) + d.ξ
invlogcdf(d::Johnson, lq::Real) = d.λ * sinh((invlogcdf(Normal(), lq) - d.γ) / d.δ) + d.ξ
invlogccdf(d::Johnson, lq::Real) = d.λ * sinh((invlogccdf(Normal(), lq) - d.γ) / d.δ) + d.ξ

# entropy(d::Johnson)
# mgf(d::Johnson)
# cf(d::Johnson)

#### Sampling

rand(rng::AbstractRNG, d::Johnson) = d.λ * sinh((rand(rng) - d.γ) / d.δ) + d.ξ

## Fitting

# function fit_mle(::Type{<:Johnson}, x::AbstractArray{T}) where T<:Real
