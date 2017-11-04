struct GenericSampler{T<:Real, S<:AbstractVector{T}} <: Sampleable{Univariate,Discrete}
    support::S
    aliastable::AliasTable

    GenericSampler{T,S}(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
        new(support, AliasTable(probs))
end

GenericSampler(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
    GenericSampler{T,S}(support, probs)

rand(s::GenericSampler) =
    (@inbounds v = s.support[rand(s.aliastable)]; v)
