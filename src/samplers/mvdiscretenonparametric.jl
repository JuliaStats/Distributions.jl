"""
    MvDiscreteNonParametricSampler(support, p)
Data structure for efficiently sampling from an arbitrary probability mass
function defined by `support` and probabilities `p`.
"""
struct MvDiscreteNonParametricSampler{T <: Real,S <: AbstractVector{<:AbstractVector{T}},A <: AliasTable} <: Sampleable{Multivariate,Discrete}
    support::S
    aliastable::A

    function MvDiscreteNonParametricSampler{T,S}(support::S, probs::AbstractVector{<:Real}
                                                ) where {T <: Real,S <: AbstractVector{<:AbstractVector{T}}}
        aliastable = AliasTable(probs)
        new{T,S,typeof(aliastable)}(support, aliastable)
    end
end

MvDiscreteNonParametricSampler(support::S, p::AbstractVector{<:Real}
                              ) where {T <: Real,S <: AbstractVector{<:AbstractVector{T}}} =
    MvDiscreteNonParametricSampler{T,S}(support, p)

# Sampling

sampler(d::MvDiscreteNonParametric) =
    MvDiscreteNonParametricSampler(support(d), probs(d))

_rand!(s::MvDiscreteNonParametricSampler, x::AbstractVector{T}) where T<:Real =
    _rand!(GLOBAL_RNG, s, x)

function _rand!(rng::AbstractRNG, d::MvDiscreteNonParametric, x::AbstractVector{T}) where T <: Real
    length(x) == length(d) || throw(DimensionMismatch("Invalid argument dimension."))
    s = d.support
    p = d.p

    n = length(p)
    draw = Base.rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i += 1]
    end
    for (j, v) in enumerate(s[i])
        x[j] = v
    end
    return x
end
