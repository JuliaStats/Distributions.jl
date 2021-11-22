struct PoissBinAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

PoissBinAliasSampler(d::PoissonBinomial) = PoissBinAliasSampler(AliasTable(d.pmf))

function rand(rng::AbstractRNG, ::Type{T}, s::PoissBinAliasSampler) where {T}
    return rand(rng, T, s.table) - 1
end
