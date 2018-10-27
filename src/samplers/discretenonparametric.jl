"""
    DiscreteNonParametricSampler(xs, ps)

Data structure for efficiently sampling from an arbitrary probability mass
function defined by support `xs` and probabilities `ps`.
"""
struct DiscreteNonParametricSampler{T<:Real, S<:AbstractVector{T}} <: Sampleable{Univariate,Discrete}
    support::S
    aliastable::AliasTable

    DiscreteNonParametricSampler{T,S}(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
        new(support, AliasTable(probs))
end

DiscreteNonParametricSampler(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
    DiscreteNonParametricSampler{T,S}(support, probs)

rand(s::DiscreteNonParametricSampler) =
    (@inbounds v = s.support[rand(s.aliastable)]; v)
