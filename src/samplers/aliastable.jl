struct AliasTable <: Sampleable{Univariate,Discrete}
    at::AliasTables.AliasTable{UInt64, Int}
    AliasTable(probs::AbstractVector{<:Real}) = new(AliasTables.AliasTable(probs))
end
ncategories(s::AliasTable) = length(s.at)
rand(rng::AbstractRNG, s::AliasTable) = rand(rng, s.at)
show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", ncategories(s))
