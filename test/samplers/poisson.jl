using Distributions
using Base.Test
import Distributions: PoissonADSampler, PoissonCountSampler

function test_poissonsampler(s, μ::Float64, ns::Int, tol::Float64)
    ub = iceil(μ * 3) + 1
    pv = Distributions.poissonpvec(μ, ub)
    @assert length(pv) == ub + 1

    cnts = zeros(Int, ub+1)
    for i = 1:ns
        x = rand(s)
        @assert x >= 0
        if x <= ub
            cnts[x + 1] += 1
        end
    end

    q = cnts ./ ns
    @test_approx_eq_eps q pv tol
end

# test cases

for μ in [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    test_poissonsampler(PoissonCountSampler(μ), μ, 10^5, 0.02)
end

for μ in [5.0, 10.0, 15.0, 20.0, 30.0]
    test_poissonsampler(PoissonADSampler(μ), μ, 10^5, 0.02)
end

