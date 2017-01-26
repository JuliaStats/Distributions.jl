immutable NoncentralF{T<:Real} <: ContinuousUnivariateDistribution
    ν1::T
    ν2::T
    λ::T

    function NoncentralF(ν1::T, ν2::T, λ::T)
        @check_args(NoncentralF, ν1 > zero(T) && ν2 > zero(T))
        @check_args(NoncentralF, λ >= zero(T))
	    new(ν1, ν2, λ)
    end
end

NoncentralF{T<:Real}(ν1::T, ν2::T, λ::T) = NoncentralF{T}(ν1, ν2, λ)
NoncentralF(ν1::Real, ν2::Real, λ::Real) = NoncentralF(promote(ν1, ν2, λ)...)
NoncentralF(ν1::Integer, ν2::Integer, λ::Integer) = NoncentralF(Float64(ν1), Float64(ν2), Float64(λ))

@distr_support NoncentralF 0.0 Inf

#### Conversions

function convert{T <: Real, S <: Real}(::Type{NoncentralF{T}}, ν1::S, ν2::S, λ::S)
    NoncentralF(T(ν1), T(ν2), T(λ))
end
function convert{T <: Real, S <: Real}(::Type{NoncentralF{T}}, d::NoncentralF{S})
    NoncentralF(T(d.ν1), T(d.ν2), T(d.λ))
end

### Parameters

params(d::NoncentralF) = (d.ν1, d.ν2, d.λ)
@inline partype{T<:Real}(d::NoncentralF{T}) = T


### Statistics

function mean{T<:Real}(d::NoncentralF{T})
    d.ν2 > 2 ? d.ν2 / (d.ν2 - 2) * (d.ν1 + d.λ) / d.ν1 : T(NaN)
end

var{T<:Real}(d::NoncentralF{T}) = d.ν2 > 4 ? 2d.ν2^2 *
		       ((d.ν1 + d.λ)^2 + (d.ν2 - 2)*(d.ν1 + 2d.λ)) /
		       (d.ν1 * (d.ν2 - 2)^2 * (d.ν2 - 4)) : T(NaN)


### Evaluation & Sampling

@_delegate_statsfuns NoncentralF nfdist ν1 ν2 λ

function rand(d::NoncentralF)
    r1 = rand(NoncentralChisq(d.ν1,d.λ)) / d.ν1
    r2 = rand(Chisq(d.ν2)) / d.ν2
    r1 / r2
end
