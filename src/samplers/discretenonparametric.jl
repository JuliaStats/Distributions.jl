"""
    DiscreteNonParametricSampler(xs, ps)

Data structure for efficiently sampling from an arbitrary probability mass
function defined by support `xs` and probabilities `ps`.
"""
struct DiscreteNonParametricSampler{T<:Real, S<:AbstractVector{T}} <: Sampleable{Univariate,Discrete}
    support::S
    aliastable::AliasTable

    DiscreteNonParametricSampler{T,S}(support::S, probs::AbstractVector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
        new(support, AliasTable(probs))
end

DiscreteNonParametricSampler(support::S, probs::AbstractVector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
    DiscreteNonParametricSampler{T,S}(support, probs)

rand(rng::AbstractRNG, s::DiscreteNonParametricSampler) =
    (@inbounds v = s.support[rand(rng, s.aliastable)]; v)

rand(s::DiscreteNonParametricSampler) = rand(Random.GLOBAL_RNG, s)
