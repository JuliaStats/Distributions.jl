using Distributions
using OffsetArrays

using Random
using Test

# #1474
@testset "median and quantile" begin
    for (a, b) in ((-8, -6), (-8, -5), (-5, 1), (-5, 2), (-2, 3), (-2, 4), (2, 4), (2, 5))
        d = DiscreteUniform(a, b)

        for x in a:b
            @test @inferred(quantile(d, cdf(d, x))) === x
        end

        @test @inferred(median(d)) === quantile(d, 1//2)
    end
end

@testset "type stability regression tests" begin
    @test @inferred(mgf(DiscreteUniform(2, 5), 0//1)) === 1.0
    @test @inferred(cf(DiscreteUniform(1, 3), 0//1)) === 1.0 + 0.0im
end

@testset "fit: array indexing (#1253)" begin
    x = shuffle(10:20)
    for data in (x, OffsetArray(x, -5:5))
        @test fit(DiscreteUniform, data) == DiscreteUniform(10, 20)
    end
end
