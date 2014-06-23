#### naive sampling 

immutable CategoricalDirectSampler <: Sampleable{Univariate,Discrete}
    prob::Vector{Float64}

    function CategoricalDirectSampler(p::Vector{Float64})
        isempty(p) && error("p is empty.")
        new(p)
    end
end
ncategories(s::CategoricalDirectSampler) = length(s.prob)

function rand(s::CategoricalDirectSampler)
    p = s.prob
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end
    return i
end


##### Alias Table #####

immutable AliasTable <: Sampleable{Univariate,Discrete}
    accept::Vector{Float64}
    alias::Vector{Int}
    isampler::RandIntSampler
end
ncategories(s::AliasTable) = length(s.accept)

function AliasTable{T<:Real}(probs::AbstractVector{T})
    n = length(probs)
    n > 0 || error("The input probability vector is empty.")
    accp = Array(Float64, n)
    alias = Array(Int, n)
    StatsBase.make_alias_table!(probs, 1.0, accp, alias)
    AliasTable(accp, alias, RandIntSampler(n))
end

rand(s::AliasTable) = 
    (i = rand(s.isampler); u = rand(); @inbounds r = u < s.accept[i] ? i : s.alias[i]; r)

show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", numcategories(s))

