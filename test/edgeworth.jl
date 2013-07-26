#

using Distributions
using Base.Test

d = EdgeworthSum(Gamma(1,1),10)
X = vec(sum(rand(d.dist,1_000_000,int(d.n)),2))
de = EmpiricalUnivariateDistribution(X)

for i = 0.01:0.01:0.99
    @test_approx_eq quantile(d,i) cquantile(d,1-i) 
    @test_approx_eq_eps cdf(d,quantile(d,i))/i 1.0 0.2
    @test_approx_eq_eps ccdf(d,cquantile(d,i))/i 1.0 0.2

    @test_approx_eq_eps quantile(d,i)/quantile(de,i) 1.0 0.02
end



