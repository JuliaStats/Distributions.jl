# Testing discrete univariate distributions

using Distributions
using Base.Test

### load reference data

immutable DiscreteRefEntry
    distr::DiscreteUnivariateDistribution
    mean::Float64
    var::Float64
    entropy::Float64
    x25::Int
    x50::Int
    x75::Int
    lp25::Float64
    lp50::Float64
    lp75::Float64
end

function DiscreteRefEntry(row::Vector)
    @assert length(row) == 10
    d = eval(parse(row[1]))
    return DiscreteRefEntry(d, row[2:10]...)
end

csvpath = joinpath(dirname(@__FILE__), "discrete_ref.csv")
table = readcsv(csvpath)

R = [DiscreteRefEntry(vec(table[i,:])) for i = 2:size(table,1)]


### check with references

function verify(e::DiscreteRefEntry)
    d = e.distr
    @test_approx_eq_eps mean(d)    e.mean    1.0e-12
    @test_approx_eq_eps var(d)     e.var     1.0e-12

    if applicable(entropy, d)
        @test_approx_eq_eps entropy(d) e.entropy 1.0e-6 * (abs(e.entropy) + 1.0)
    end

    @test int(quantile(d, 0.25)) == e.x25
    @test int(quantile(d, 0.50)) == e.x50
    @test int(quantile(d, 0.75)) == e.x75

    @test_approx_eq_eps logpdf(d, e.x25) e.lp25 1.0e-12
    @test_approx_eq_eps logpdf(d, e.x50) e.lp50 1.0e-12
    @test_approx_eq_eps logpdf(d, e.x75) e.lp75 1.0e-12
end

n_tsamples = 10^6

for rentry in R
    println("    testing $(rentry.distr)")
    verify(rentry)
    test_distr(rentry.distr, n_tsamples)
end

