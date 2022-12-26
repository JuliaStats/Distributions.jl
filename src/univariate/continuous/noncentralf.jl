"""
    NoncentralF(ν1, ν2, λ)

*Noncentral F-distribution* with `ν1 > 0` and `ν2 > 0` degrees of freedom and noncentrality parameter `λ >= 0`.
"""
struct NoncentralF{T<:Real} <: ContinuousUnivariateDistribution
    ν1::T
    ν2::T
    λ::T
    NoncentralF{T}(ν1::T, ν2::T, λ::T) where {T} = new{T}(ν1, ν2, λ)
end

function NoncentralF(ν1::T, ν2::T, λ::T; check_args::Bool=true) where {T <: Real}
    @check_args NoncentralF (ν1, ν1 > zero(ν1)) (ν2, ν2 > zero(ν2)) (λ, λ >= zero(λ))
    return NoncentralF{T}(ν1, ν2, λ)
end

NoncentralF(ν1::Real, ν2::Real, λ::Real; check_args::Bool=true) = NoncentralF(promote(ν1, ν2, λ)...; check_args=check_args)
NoncentralF(ν1::Integer, ν2::Integer, λ::Integer; check_args::Bool=true) = NoncentralF(float(ν1), float(ν2), float(λ); check_args=check_args)

@distr_support NoncentralF 0.0 Inf

#### Conversions

function convert(::Type{NoncentralF{T}}, ν1::S, ν2::S, λ::S) where {T <: Real, S <: Real}
    NoncentralF(T(ν1), T(ν2), T(λ))
end
function Base.convert(::Type{NoncentralF{T}}, d::NoncentralF) where {T<:Real}
    NoncentralF{T}(T(d.ν1), T(d.ν2), T(d.λ))
end
Base.convert(::Type{NoncentralF{T}}, d::NoncentralF{T}) where {T<:Real} = d

### Parameters

params(d::NoncentralF) = (d.ν1, d.ν2, d.λ)
partype(::NoncentralF{T}) where {T} = T

### Statistics

function mean(d::NoncentralF{T}) where {T<:Real}
    d.ν2 > 2 ? d.ν2 / (d.ν2 - 2) * (d.ν1 + d.λ) / d.ν1 : T(NaN)
end

var(d::NoncentralF{T}) where {T<:Real} = d.ν2 > 4 ? 2d.ν2^2 *
               ((d.ν1 + d.λ)^2 + (d.ν2 - 2)*(d.ν1 + 2d.λ)) /
               (d.ν1 * (d.ν2 - 2)^2 * (d.ν2 - 4)) : T(NaN)


### Evaluation & Sampling

@_delegate_statsfuns NoncentralF nfdist ν1 ν2 λ

function rand(rng::AbstractRNG, d::NoncentralF)
    r1 = rand(rng, NoncentralChisq(d.ν1,d.λ)) / d.ν1
    r2 = rand(rng, Chisq(d.ν2)) / d.ν2
    r1 / r2
end

# TODO: remove RFunctions dependency once NoncentralChisq has its removed
@rand_rdist(NoncentralF)
function rand(d::NoncentralF)
    r1 = rand(NoncentralChisq(d.ν1,d.λ)) / d.ν1
    r2 = rand(Chisq(d.ν2)) / d.ν2
    r1 / r2
end
