"""
    NormalCanon(η, λ)

Canonical parametrisation of the Normal distribution with canonical parameters `η` and `λ`.

The two *canonical parameters* of a normal distribution ``\\mathcal{N}(\\mu, \\sigma^2)`` with mean ``\\mu`` and
standard deviation ``\\sigma`` are ``\\eta = \\sigma^{-2} \\mu`` and ``\\lambda = \\sigma^{-2}``.
"""
struct NormalCanon{T<:Real} <: ContinuousUnivariateDistribution
    η::T       # σ^(-2) * μ
    λ::T       # σ^(-2)
    μ::T       # μ

    function NormalCanon{T}(η, λ; check_args::Bool=true) where T
        @check_args NormalCanon (λ, λ > zero(λ))
        new{T}(η, λ, η / λ)
    end
end

NormalCanon(η::T, λ::T; check_args::Bool=true) where {T<:Real} = NormalCanon{typeof(η/λ)}(η, λ; check_args=check_args)
NormalCanon(η::Real, λ::Real; check_args::Bool=true) = NormalCanon(promote(η, λ)...; check_args=check_args)
NormalCanon(η::Integer, λ::Integer; check_args::Bool=true) = NormalCanon(float(η), float(λ); check_args=check_args)
NormalCanon() = NormalCanon{Float64}(0.0, 1.0; check_args=false)

@distr_support NormalCanon -Inf Inf

#### Type Conversions
convert(::Type{NormalCanon{T}}, η::S, λ::S) where {T <: Real, S <: Real} = NormalCanon(T(η), T(λ))
Base.convert(::Type{NormalCanon{T}}, d::NormalCanon) where {T<:Real} = NormalCanon{T}(T(d.η), T(d.λ); check_args=false)
Base.convert(::Type{NormalCanon{T}}, d::NormalCanon{T}) where {T<:Real} = d

## conversion between Normal and NormalCanon

convert(::Type{Normal}, d::NormalCanon) = Normal(d.μ, 1 / sqrt(d.λ))
convert(::Type{NormalCanon}, d::Normal) = (λ = 1 / d.σ^2; NormalCanon(λ * d.μ, λ))
meanform(d::NormalCanon) = convert(Normal, d)
canonform(d::Normal) = convert(NormalCanon, d)


#### Parameters

params(d::NormalCanon) = (d.η, d.λ)
@inline partype(d::NormalCanon{T}) where {T<:Real} = T

#### Statistics

mean(d::NormalCanon) = d.μ
median(d::NormalCanon) = mean(d)
mode(d::NormalCanon) = mean(d)

skewness(d::NormalCanon{T}) where {T<:Real} = zero(T)
kurtosis(d::NormalCanon{T}) where {T<:Real} = zero(T)

var(d::NormalCanon) = 1 / d.λ
std(d::NormalCanon) = sqrt(var(d))

entropy(d::NormalCanon) = (-log(d.λ) + log2π + 1) / 2

location(d::NormalCanon) = mean(d)
scale(d::NormalCanon) = std(d)

function kldivergence(p::NormalCanon, q::NormalCanon)
    μp = mean(p)
    μq = mean(q)
    σ²p_over_σ²q = q.λ / p.λ
    return (abs2(μp - μq) * q.λ - logmxp1(σ²p_over_σ²q)) / 2
end

#### Evaluation

pdf(d::NormalCanon, x::Real) = (sqrt(d.λ) / sqrt2π) * exp(-d.λ * abs2(x - d.μ)/2)
logpdf(d::NormalCanon, x::Real) = (log(d.λ) - log2π - d.λ * abs2(x - d.μ))/2

zval(d::NormalCanon, x::Real) = (x - d.μ) * sqrt(d.λ)
xval(d::NormalCanon, z::Real) = d.μ + z / sqrt(d.λ)

cdf(d::NormalCanon, x::Real) = normcdf(zval(d,x))
ccdf(d::NormalCanon, x::Real) = normccdf(zval(d,x))
logcdf(d::NormalCanon, x::Real) = normlogcdf(zval(d,x))
logccdf(d::NormalCanon, x::Real) = normlogccdf(zval(d,x))

quantile(d::NormalCanon, p::Real) = xval(d, norminvcdf(p))
cquantile(d::NormalCanon, p::Real) = xval(d, norminvccdf(p))
invlogcdf(d::NormalCanon, lp::Real) = xval(d, norminvlogcdf(lp))
invlogccdf(d::NormalCanon, lp::Real) = xval(d, norminvlogccdf(lp))


#### Sampling

rand(rng::AbstractRNG, cf::NormalCanon) = cf.μ + randn(rng) / sqrt(cf.λ)

#### Affine transformations

function Base.:+(d::NormalCanon, c::Real)
    η, λ = params(d)
    return NormalCanon(η + c * λ, λ)
end
Base.:*(c::Real, d::NormalCanon) = NormalCanon(d.η / c, d.λ / c^2)
