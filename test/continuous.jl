# Testing continuous univariate distributions

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

immutable ContinuousRefEntry
    distr::ContinuousUnivariateDistribution
    mean::Float64
    var::Float64
    entropy::Float64
    x25::Float64
    x50::Float64
    x75::Float64
    lp25::Float64
    lp50::Float64
    lp75::Float64
end

function ContinuousRefEntry(row::Vector)
    @assert length(row) == 10
    d = eval(parse(row[1]))
    return ContinuousRefEntry(d, row[2:10]...)
end

csvpath = joinpath(dirname(@__FILE__), "continuous_ref.csv")
table = readcsv(csvpath)

R = [ContinuousRefEntry(vec(table[i,:])) for i = 2:size(table,1)]


### check with references

function verify(e::ContinuousRefEntry)
    d = e.distr

    if isfinite(e.mean)
        @test_approx_eq_eps mean(d)    e.mean    1.0e-12
    end

    if isfinite(e.var)
        @test_approx_eq_eps var(d)     e.var     1.0e-12
    end

    if isfinite(e.entropy) && applicable(entropy, d)
        @test_approx_eq_eps entropy(d) e.entropy 1.0e-6 * (abs(e.entropy) + 1.0)
    end

    @test_approx_eq_eps quantile(d, 0.25) e.x25 5.0e-9
    @test_approx_eq_eps quantile(d, 0.50) e.x50 5.0e-9
    @test_approx_eq_eps quantile(d, 0.75) e.x75 5.0e-9

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

# additional distributions that have no direct counterparts in scipy
println("    -----")

for distr in [   
    Frechet(0.5, 1.0),
    Frechet(3.0, 1.0),
    Frechet(20.0, 1.0),
    Frechet(120.0, 1.0),
    Frechet(0.5, 2.0),
    Frechet(3.0, 2.0),
    InverseGaussian(1.0, 1.0),
    InverseGaussian(2.0, 7.0),
    Levy(0.0, 1.0),
    Levy(2.0, 8.0),
    Levy(3.0, 3.0),
    LogNormal(0.0, 1.0),
    LogNormal(0.0, 2.0),
    LogNormal(3.0, 0.5),
    LogNormal(3.0, 1.0),
    LogNormal(3.0, 2.0),
    NoncentralBeta(2,2,0),
    NoncentralBeta(2,6,5),
    NoncentralChisq(2,2),
    NoncentralChisq(2,5),
    NoncentralF(2,2,2),
    NoncentralF(8,10,5),
    NoncentralT(2,2),
    NoncentralT(10,2) ]

    println("    testing $(distr)")
    test_distr(distr, n_tsamples)
 end

