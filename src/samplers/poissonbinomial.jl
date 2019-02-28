struct PoissBinAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

PoissBinAliasSampler(d::PoissonBinomial) = PoissBinAliasSampler(AliasTable(d.pmf))

rand(rng::AbstractRNG, s::PoissBinAliasSampler) = rand(rng, s.table) - 1
