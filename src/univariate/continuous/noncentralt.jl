"""
    NoncentralT(ν, λ)

*Noncentral Student's t-distribution* with `v > 0` degrees of freedom and noncentrality parameter `λ`.
"""
struct NoncentralT{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    λ::T
    NoncentralT{T}(ν::T, λ::T) where {T} = new{T}(ν, λ)
end

function NoncentralT(ν::T, λ::T; check_args::Bool=true) where {T <: Real}
    @check_args NoncentralT (ν, ν > zero(ν))
    return NoncentralT{T}(ν, λ)
end

NoncentralT(ν::Real, λ::Real; check_args::Bool=true) = NoncentralT(promote(ν, λ)...; check_args=check_args)
NoncentralT(ν::Integer, λ::Integer; check_args::Bool=true) = NoncentralT(float(ν), float(λ); check_args=check_args)

@distr_support NoncentralT -Inf Inf

### Conversions
convert(::Type{NoncentralT{T}}, ν::S, λ::S) where {T <: Real, S <: Real} = NoncentralT(T(ν), T(λ))
Base.convert(::Type{NoncentralT{T}}, d::NoncentralT) where {T<:Real} = NoncentralT(T(d.ν), T(d.λ))
Base.convert(::Type{NoncentralT{T}}, d::NoncentralT{T}) where {T<:Real} = d

### Parameters

params(d::NoncentralT) = (d.ν, d.λ)
partype(::NoncentralT{T}) where {T} = T


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
