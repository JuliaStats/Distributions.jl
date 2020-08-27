struct AliasTable <: Sampleable{Univariate,Discrete}
    accept::Vector{Float64}
    alias::Vector{Int}
end
ncategories(s::AliasTable) = length(s.alias)

function AliasTable(probs::AbstractVector)
    n = length(probs)
    n > 0 || throw(ArgumentError("The input probability vector is empty."))
    accp = Vector{Float64}(undef, n)
    alias = Vector{Int}(undef, n)
    StatsBase.make_alias_table!(probs, 1.0, accp, alias)
    AliasTable(accp, alias)
end

function rand(rng::AbstractRNG, s::AliasTable)
    i = rand(rng, 1:length(s.alias)) % Int
    u = rand(rng)
    @inbounds r = u < s.accept[i] ? i : s.alias[i]
    r
end

show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", ncategories(s))
