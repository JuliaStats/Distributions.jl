using Distributions
if VERSION >= v"0.7.0-DEV"
    using Test
else
    using Base.Test
end

d = Normal()
@test Distributions.quantile_newton(d, 0.5) == quantile(d, 0.5)
@test Distributions.cquantile_newton(d, 0.5) == cquantile(d, 0.5)
