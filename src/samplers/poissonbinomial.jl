struct PoissBinAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

PoissBinAliasSampler(d::PoissonBinomial) = PoissBinAliasSampler(AliasTable(d.pmf))

rand(s::PoissBinAliasSampler) = rand(s.table) - 1

