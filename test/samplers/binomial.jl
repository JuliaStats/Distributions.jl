using Distributions
using Base.Test
import Distributions: BinomialGeomSampler, BinomialTPESampler, BinomialPolySampler, BinomialAliasSampler

function test_binomsampler(s, params::(Int, Float64), ns::Int, tol::Float64)
    # s: the sampler to be tested
    # params: the binomial parameters, in the form of (n, p)
    # n: the number of samples to be generated
    # tol: the tolerance of deviation

    n, p = params
    if p == 0.0 || n == 0
        for i=1:ns
            @assert rand(s) == 0
        end
    elseif p == 1.0
        for i=1:ns
            @assert rand(s) == n
        end
    else
        cnts = zeros(Int, n+1)
        for i=1:ns
            x = rand(s)
            @assert 0 <= x <= n
            cnts[x+1] += 1
        end
        pv = Distributions.binompvec(n, p)
        pr = cnts ./ ns
        @test_approx_eq_eps pr pv tol
    end
end

# test cases

for params in [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6)]
    test_binomsampler(BinomialGeomSampler(params...), params, 10^5, 0.015)
end

for params in [(40, 0.5), (100, 0.4), (300, 0.6)]
    test_binomsampler(BinomialTPESampler(params...), params, 10^6, 0.015)
end

for params in [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), 
               (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6), 
               (40, 0.5), (100, 0.4), (300, 0.6)]
    test_binomsampler(BinomialPolySampler(params...), params, 10^6, 0.015)
end

for params in [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), 
               (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6), 
               (40, 0.5), (100, 0.4), (300, 0.6)]
    test_binomsampler(BinomialAliasSampler(params...), params, 10^6, 0.015)
end
