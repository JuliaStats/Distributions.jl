using Distributions
using Base.Test

d = Kolmogorov()
# compare to Smirnov's tabulated values
@test round(cdf(d,0.28),6) == .000_001
@test round(cdf(d,0.50),6) == .036_055
@test round(cdf(d,0.99),6) == .719_126
@test round(cdf(d,1.00),6) == .730_000
@test round(cdf(d,1.01),6) == .740_566
# @test round(cdf(d,1.04),6) == .770_434 # table value appears to be wrong
@test round(cdf(d,1.50),6) == .977_782
@test round(cdf(d,2.00),6) == .999_329
@test round(cdf(d,2.50),7) == .999_992_5
@test round(cdf(d,3.00),8) == .999_999_97
