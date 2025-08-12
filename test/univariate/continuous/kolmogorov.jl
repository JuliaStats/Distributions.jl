using Distributions
using Test


d = Kolmogorov()
# compare to Smirnov's tabulated values
@test round(cdf(d, 0.28), digits = 6) == 0.000_001
@test round(cdf(d, 0.5), digits = 6) == 0.036_055
@test round(cdf(d, 0.99), digits = 6) == 0.719_126
@test round(cdf(d, 1.0), digits = 6) == 0.730_000
@test round(cdf(d, 1.01), digits = 6) == 0.740_566
# @test round(cdf(d,1.04),digits=6) == .770_434 # table value appears to be wrong
@test round(cdf(d, 1.5), digits = 6) == 0.977_782
@test round(cdf(d, 2.0), digits = 6) == 0.999_329
@test round(cdf(d, 2.5), digits = 7) == 0.999_992_5
@test round(cdf(d, 3.0), digits = 8) == 0.999_999_97
