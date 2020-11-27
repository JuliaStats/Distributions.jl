"""
    NoncentralBeta(α, β, λ)
"""
struct NoncentralBeta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    λ::T
    NoncentralBeta{T}(α::T, β::T, λ::T) where {T} = new{T}(α, β, λ)
end

function NoncentralBeta(α::T, β::T, λ::T; check_args=true) where {T <: Real}
    check_args && @check_args(NoncentralBeta, α > zero(α) && β > zero(β) && λ >= zero(λ))
    return NoncentralBeta{T}(α, β, λ)
end

NoncentralBeta(α::Real, β::Real, λ::Real) = NoncentralBeta(promote(α, β, λ)...)
NoncentralBeta(α::Integer, β::Integer, λ::Integer) = NoncentralBeta(float(α), float(β), float(λ))

@distr_support NoncentralBeta 0.0 1.0

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
