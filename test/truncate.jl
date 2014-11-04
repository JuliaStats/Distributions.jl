using Distributions
using Base.Test

n_tsamples = 10^6

for (pa0, lb, ub) in [ 
    ((0, 1), -2, 2),
    ((3, 10), 7, 8),
    ((27, 3), 0, Inf),
    ((-5, 1), -Inf, -10),
    ((1.8, 1.2), -Inf, 0) 
    ]

    d = Truncated(Normal(pa0...), lb, ub)
    println("    testing $d")

    @test d.lower == lb
    @test d.upper == ub
    @test minimum(d) == lb
    @test maximum(d) == ub

    test_distr(d, n_tsamples)
end

