# Testing discrete univariate distributions

using Distributions
using Base.Test

### load reference data
#
#   Note
#   -------
#   To generate the reference data:
#   (1) make sure that python, numpy, and scipy are installed in your system
#   (2) enter the sub-directory test
#   (3) run: python discrete_ref.py > discrete_ref.csv
#  
#   For most cases, you don't have. You only need to run this when you
#   implement a new distribution and want to add new test cases, then
#   you should add the new test cases to discrete_ref.py and run this
#   procedure to update the reference data.
#

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

# for Categorical (scipy.stats no counterpart)

for distr in [
    Categorical([0.1, 0.9]),
    Categorical([0.5, 0.5]),
    Categorical([0.9, 0.1]), 
    Categorical([0.2, 0.5, 0.3])]

    @test minimum(distr) == 1
    @test maximum(distr) == ncategories(distr)
    for i = 1:distr.K
        @test pdf(distr, i) == distr.prob[i]
    end

    println("    testing $(distr)")
    test_distr(distr, n_tsamples)
end

