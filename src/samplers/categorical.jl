#### naive sampling

struct CategoricalDirectSampler <: Sampleable{Univariate,Discrete}
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


#if Base.VERSION < v"0.7"
    struct AliasTable <: Sampleable{Univariate,Discrete}
        accept::Vector{Float64}
        alias::Vector{Int}
        isampler::RangeGeneratorInt{Int,UInt}
    end
#else
#    struct AliasTable <: Sampleable{Univariate,Discrete}
#        accept::Vector{Float64}
#        alias::Vector{Int}
#        isampler::SamplerRangeInt{Int,UInt}
#    end
#end
ncategories(s::AliasTable) = length(s.accept)

function AliasTable(probs::AbstractVector{T}) where T<:Real
    n = length(probs)
    n > 0 || error("The input probability vector is empty.")
    accp = Vector{Float64}(n)
    alias = Vector{Int}(n)
    StatsBase.make_alias_table!(probs, 1.0, accp, alias)
    AliasTable(accp, alias, Base.Random.RangeGenerator(1:n))
end

function rand(rng::AbstractRNG, s::AliasTable)
    i = rand(GLOBAL_RNG, s.isampler) % Int
    u = rand()
    @inbounds r = u < s.accept[i] ? i : s.alias[i]
    r
end
rand(s::AliasTable) = rand(Base.Random.GLOBAL_RNG, s)

show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", ncategories(s))

