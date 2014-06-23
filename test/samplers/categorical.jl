using Distributions
using Base.Test
import Distributions: CategoricalDirectSampler, AliasTable

function test_categoricalsampler(s, p::Vector{Float64}, n::Int, tol::Float64)
    # s: the sampler to be tested
    # p: the reference probability vector
    # n: the number of samples to be generated
    # tol: maximal allowable deviation 

    K = length(p)
    @assert ncategories(s) == K

    cnts = zeros(Int, K)

    for i = 1:n
        k = rand(s)
        @assert 1 <= k <= K
        cnts[k] += 1
    end

    q = cnts ./ n
    # println(Base.maxabs(p - q))
    @test_approx_eq_eps p q tol
end

@test_throws ErrorException CategoricalDirectSampler(Float64[])
@test_throws ErrorException AliasTable(Float64[])


for S in [CategoricalDirectSampler, AliasTable]
    s = S([1.0])
    for i = 1:10
        @test rand(s) == 1
    end
    for p in {[0.3, 0.7], [0.2, 0.3, 0.4, 0.1]}
        test_categoricalsampler(S(p), p, 10^5, 0.015)
    end
end

