struct GenericSampler{T<:Real} <: Sampleable{Univariate,Discrete}
    vals::Vector{T}
    aliastable::AliasTable

    GenericSampler{T}(vals::Vector{T}, probs::Vector{<:Real}) where T<:Real =
        new(vals, AliasTable(probs))
end

GenericSampler(vals::Vector{T}, probs::Vector{<:Real}) where T<:Real =
    GenericSampler{T}(vals, probs)

sampler(d::Generic) =
    GenericSampler(d.vals, d.probs)

rand(s::GenericSampler) =
    (@inbounds v = s.vals[rand(s.aliastable)]; v)
