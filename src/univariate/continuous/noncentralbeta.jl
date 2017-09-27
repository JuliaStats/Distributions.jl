"""
    NoncentralBeta(α, β, λ)
"""
struct NoncentralBeta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    λ::T

    function NoncentralBeta{T}(α::T, β::T, λ::T) where T
        @check_args(NoncentralBeta, α > zero(α) && β > zero(β))
        @check_args(NoncentralBeta, λ >= zero(λ))
        new{T}(α, β, λ)
    end
end

NoncentralBeta(α::T, β::T, λ::T) where {T<:Real} = NoncentralBeta{T}(α, β, λ)
NoncentralBeta(α::Real, β::Real, λ::Real) = NoncentralBeta(promote(α, β, λ)...)
NoncentralBeta(α::Integer, β::Integer, λ::Integer) = NoncentralBeta(Float64(α), Float64(β), Float64(λ))

@distr_support NoncentralBeta 0.0 1.0


### Parameters

params(d::NoncentralBeta) = (d.α, d.β, d.λ)
@inline partype(d::NoncentralBeta{T}) where {T<:Real} = T


### Evaluation & Sampling

# TODO: add mean and var

@_delegate_statsfuns NoncentralBeta nbeta α β λ

function rand(d::NoncentralBeta)
    a = rand(NoncentralChisq(2d.α, d.β))
    b = rand(Chisq(2d.β))
    a / (a + b)
end
