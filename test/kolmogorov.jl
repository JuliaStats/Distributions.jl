using  Distributions
using Test


d = Kolmogorov()
# compare to Smirnov's tabulated values
@test round(cdf(d,0.28), digits=6) == .000_001
@test round(cdf(d,0.50), digits=6) == .036_055
@test round(cdf(d,0.99), digits=6) == .719_126
@test round(cdf(d,1.00), digits=6) == .730_000
@test round(cdf(d,1.01), digits=6) == .740_566
# @test round(cdf(d,1.04),digits=6) == .770_434 # table value appears to be wrong
@test round(cdf(d,1.50), digits=6) == .977_782
@test round(cdf(d,2.00), digits=6) == .999_329
@test round(cdf(d,2.50), digits=7) == .999_992_5
@test round(cdf(d,3.00), digits=8) == .999_999_97
