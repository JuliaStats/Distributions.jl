using Distributions
using Base.Test
import Distributions: GammaMTSampler

gamcdf(x::Float64, α::Float64, s::Float64) = 
    ccall(("pgamma", "libRmath-julia"), 
        Float64, (Float64, Float64, Float64, Int32, Int32), 
        x, α, s, 1, 0)


function test_gamsampler(s, params::(Float64, Float64), ns::Int, tol::Float64)
    # divide bins
    α, scale = params
    step = scale * 0.5
    edges = step:step:(10*scale)
    nbins = length(edges)

    # compute the portion of each bin
    pv = zeros(nbins)
    a = 0.0
    for i = 1:nbins
        apre = a
        a = gamcdf(edges[i], α, scale)
        pv[i] = a - apre
    end
    
    # sampling
    cnts = zeros(Int, nbins)
    for i = 1:ns
        x = rand(s)
        k = ifloor(x / step) + 1
        if k <= nbins
            cnts[k] += 1
        end
    end
    q = cnts ./ ns
    
    @test_approx_eq_eps q pv tol
end

# test cases

for params in [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (0.5, 1.0), 
               (1.0, 2.0), (3.0, 2.0), (0.5, 2.0)]

    test_gamsampler(GammaMTSampler(params...), params, 10^5, 0.015)
end

