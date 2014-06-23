using Distributions
using Base.Test
import Distributions: ExponentialSampler

expcdf(x::Float64, scale::Float64) = 1.0 - exp(-x/scale)

function test_expsampler(s, scale::Float64, ns::Int, tol::Float64)
    # divide bins
    edges = 0.2:0.2:(3*scale)
    nbins = length(edges)

    # compute the portion of each bin
    pv = zeros(nbins)
    a = 0.0
    for i = 1:nbins
        apre = a
        a = expcdf(edges[i], scale)
        pv[i] = a - apre
    end
    
    # sampling
    cnts = zeros(Int, nbins)
    for i = 1:ns
        x = rand(s)
        k = ifloor(x / 0.2) + 1
        if k <= nbins
            cnts[k] += 1
        end
    end
    q = cnts ./ ns

    @test_approx_eq_eps q pv tol
end

# test cases

test_expsampler(ExponentialSampler(1.0), 1.0, 10^5, 0.01)
test_expsampler(ExponentialSampler(2.0), 2.0, 10^5, 0.01)
