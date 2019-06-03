"""
    NoncentralT(ν, λ)
"""
struct NoncentralT{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    λ::T

    function NoncentralT{T}(ν::T, λ::T) where T
        @check_args(NoncentralT, ν > zero(ν))
        @check_args(NoncentralT, isreal(λ))
        new{T}(ν, λ)
    end
end

NoncentralT(ν::T, λ::T) where {T<:Real} = NoncentralT{T}(ν, λ)
NoncentralT(ν::Real, λ::Real) = NoncentralT(promote(ν, λ)...)
NoncentralT(ν::Integer, λ::Integer) = NoncentralT(Float64(ν), Float64(λ))

@distr_support NoncentralT -Inf Inf

### Conversions
convert(::Type{NoncentralT{T}}, ν::S, λ::S) where {T <: Real, S <: Real} = NoncentralT(T(ν), T(λ))
convert(::Type{NoncentralT{T}}, d::NoncentralT{S}) where {T <: Real, S <: Real} = NoncentralT(T(d.ν), T(d.λ))

### Parameters

params(d::NoncentralT) = (d.ν, d.λ)
@inline partype(d::NoncentralT{T}) where {T<:Real} = T


### Statistics

function mean(d::NoncentralT{T}) where T<:Real
    if d.ν > 1
        isinf(d.ν) ? d.λ :
        sqrt(d.ν/2) * d.λ * gamma((d.ν - 1)/2) / gamma(d.ν/2)
    else
        T(NaN)
    end
end

function var(d::NoncentralT{T}) where T<:Real
    d.ν > 2 ? d.ν*(1 + d.λ^2) / (d.ν - 2) - mean(d)^2 : T(NaN)
end

### Evaluation & Sampling

@_delegate_statsfuns NoncentralT ntdist ν λ

## sampling
function rand(rng::AbstractRNG, d::NoncentralT)
    ν = d.ν
    z = randn(rng)
    v = rand(rng, Chisq(ν))
    (z+d.λ)/sqrt(v/ν)
end
