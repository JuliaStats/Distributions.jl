struct AliasTable <: Sampleable{Univariate,Discrete}
    at::AliasTables.AliasTable{UInt64, Int}
    AliasTable(probs::AbstractVector{<:Real}) = new(AliasTables.AliasTable(probs))
end
ncategories(s::AliasTable) = length(s.at)

rand(rng::AbstractRNG, s::AliasTable) = rand(rng, s.at)
function _rand!(rng::AbstractRNG, s::AliasTable, x::AbstractArray{<:Real})
    rand!(rng, x, s.at)
    return x
end

show(io::IO, s::AliasTable) = print(io, "AliasTable with ", ncategories(s), " entries")
