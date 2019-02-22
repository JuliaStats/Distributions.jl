struct PoissBinAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

PoissBinAliasSampler(d::PoissonBinomial) = PoissBinAliasSampler(AliasTable(d.pmf))

_rand(rng::AbstractRNG, s::PoissBinAliasSampler) = _rand(rng, s.table) - 1
