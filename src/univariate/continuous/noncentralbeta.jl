"""
    NoncentralBeta(α, β, λ)

*Noncentral Beta distribution* with shape parameters `α > 0` and `β > 0` and noncentrality parameter `λ >= 0`.
"""
struct NoncentralBeta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    λ::T
    NoncentralBeta{T}(α::T, β::T, λ::T) where {T} = new{T}(α, β, λ)
end

function NoncentralBeta(α::T, β::T, λ::T; check_args::Bool=true) where {T <: Real}
    @check_args NoncentralBeta (α, α > zero(α)) (β, β > zero(β)) (λ, λ >= zero(λ))
    return NoncentralBeta{T}(α, β, λ)
end

NoncentralBeta(α::Real, β::Real, λ::Real; check_args::Bool=true) = NoncentralBeta(promote(α, β, λ)...; check_args=check_args)
NoncentralBeta(α::Integer, β::Integer, λ::Integer; check_args::Bool=true) = NoncentralBeta(float(α), float(β), float(λ); check_args=check_args)

@distr_support NoncentralBeta 0.0 1.0

#### Conversions

function Base.convert(::Type{NoncentralBeta{T}}, d::NoncentralBeta) where {T<:Real}
    NoncentralBeta{T}(T(d.α), T(d.β), T(d.λ))
end
Base.convert(::Type{NoncentralBeta{T}}, d::NoncentralBeta{T}) where {T<:Real} = d

### Parameters

params(d::NoncentralBeta) = (d.α, d.β, d.λ)
partype(::NoncentralBeta{T}) where {T} = T


### Evaluation & Sampling

# TODO: add mean and var

@_delegate_statsfuns NoncentralBeta nbeta α β λ

# TODO: remove RFunctions dependency once NoncentralChisq has its removed
@rand_rdist(NoncentralBeta)

function rand(d::NoncentralBeta)
    β = d.β
    a = rand(NoncentralChisq(2d.α, β))
    b = rand(Chisq(2β))
    a / (a + b)
end

function rand(rng::AbstractRNG, d::NoncentralBeta)
    β = d.β
    a = rand(rng, NoncentralChisq(2d.α, β))
    b = rand(rng, Chisq(2β))
    a / (a + b)
end
