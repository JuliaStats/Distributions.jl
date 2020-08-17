struct AliasTable{N} <: Sampleable{Univariate,Discrete}
    accept::NTuple{N,Float64}
    alias::NTuple{N,Int}
    sample_space::UnitRange{Int64}
end
ncategories(s::AliasTable{N}) where N = N

function AliasTable(probs::AbstractVector{T}) where T<:Real
    N = length(probs)
    N > 0 || throw(ArgumentError("The input probability vector is empty."))
    accp = Vector{Float64}(undef, N)
    alias = Vector{Int}(undef, N)
    StatsBase.make_alias_table!(probs, 1.0, accp, alias)
    AliasTable{N}(tuple(accp...), tuple(alias...), 1:N)
end

function rand(rng::AbstractRNG, s::AliasTable)
    i = rand(rng, s.sample_space) % Int
    u = rand(rng)
    @inbounds r = u < s.accept[i] ? i : s.alias[i]
    r
end

show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", ncategories(s))
