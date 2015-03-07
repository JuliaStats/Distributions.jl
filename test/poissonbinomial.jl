using Distributions
using Base.Test

# Test the special base where PoissonBinomial distribution reduces 
# to Binomial distribution
for (p, n) in [(0.8, 6), (0.5, 10), (0.04, 20)]

    d = PoissonBinomial(fill(p, n))
    dref = Binomial(n, p)
    println("   testing PoissonBinomial p=$p, n=$n")

    @test isa(d, PoissonBinomial)
    @test minimum(d) == 0
    @test maximum(d) == n
    @test ntrials(d) == n
    @test_approx_eq entropy(d) entropy(dref)
    @test_approx_eq median(d) median(dref)
    @test_approx_eq mean(d) mean(dref)
    @test_approx_eq var(d) var(dref)
    @test_approx_eq kurtosis(d) kurtosis(dref)
    for i=0:n
        @test_approx_eq_eps cdf(d, i) cdf(dref, i) 1e-15
        @test_approx_eq_eps pdf(d, i) pdf(dref, i) 1e-15
    end


end
