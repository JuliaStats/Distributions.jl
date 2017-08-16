struct PoissBinAliasSampler <: Sampleable{Univariate,Discrete}
    table::AliasTable
end

function PoissBinAliasSampler(p::AbstractVector)
    n = length(p)
    pv = poissonbinomial_pdf_fft(p)
    alias = Vector{Int}(n+1)
    StatsBase.make_alias_table!(pv, 1.0, pv, alias)
    BinomialAliasSampler(AliasTable(pv, alias, RangeGenerator(1:n+1)))
end

function PoissBinAliasSampler(d::PoissonBinomial)
    n = length(d.p)
    alias = Vector{Int}(n+1)
    pv = Vector{Float64}(n+1)
    StatsBase.make_alias_table!(d.pmf, 1.0, pv, alias)
    BinomialAliasSampler(AliasTable(pv, alias, RangeGenerator(1:n+1)))
end

rand(s::PoissBinAliasSampler) = rand(s.table) - 1

