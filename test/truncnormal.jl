using Test
using Distributions, Random

rng = MersenneTwister(123)

@testset "Truncated normal, mean and variance" begin
    d = Distributions.TruncatedNormal(0,1,100,115); @test mean(d) ≈ 100.00999800099926070518490239457545847490332879043
    d = Distributions.TruncatedNormal(0,1,50,70); @test var(d) ≈ 0.00039904318680389954790992722653605933053648912703600
    d = Distributions.TruncatedNormal(-2,3,50,70); @test mean(d) ≈ 50.171943499898757645751683644632860837133138152489
    d = Distributions.TruncatedNormal(-2,3,50,70); @test var(d) ≈ 0.029373438107168350377591231295634273607812172191712
end
@testset "Truncated normal $trunc" for trunc in [TruncatedNormal(0, 1, -2, 2),
                                                 Truncated(Normal(0, 1), -2, 2)]
    @testset "Truncated normal $trunc with $func" for func in
        [(r = rand, r! = rand!),
         (r = ((d, n) -> rand(rng, d, n)), r! = ((d, X) -> rand!(rng, d, X)))]
        repeats = 1000000
        
        @test abs(mean(func.r(trunc, repeats))) < 0.01
        @test abs(median(func.r(trunc, repeats))) < 0.01
        @test abs(var(func.r(trunc, repeats)) - var(trunc)) < 0.01

        X = Matrix{Float64}(undef, 1000, 1000)
        func.r!(trunc, X)
        @test abs(mean(X)) < 0.01
        @test abs(median(X)) < 0.01
        @test abs(var(X) - var(trunc)) < 0.01
    end
end
