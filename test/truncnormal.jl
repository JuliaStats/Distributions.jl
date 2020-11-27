using Test
using Distributions, Random

rng = MersenneTwister(123)

@testset "Truncated normal, mean and variance" begin
    @test mean(truncated(Normal(0,1),100,115)) ≈ 100.00999800099926070518490239457545847490332879043
    @test mean(truncated(Normal(-2,3),50,70)) ≈ 50.171943499898757645751683644632860837133138152489
    @test mean(truncated(Normal(0,2),-100,0)) ≈ -1.59576912160573071175978423973752747390343452465973
    @test mean(truncated(Normal(0,1),-Inf,Inf)) == 0
    @test mean(truncated(Normal(0,1),0,+Inf)) ≈ +√(2/π)
    @test mean(truncated(Normal(0,1),-Inf,0)) ≈ -√(2/π)
    @test var(truncated(Normal(0,1),50,70)) ≈ 0.00039904318680389954790992722653605933053648912703600
    @test var(truncated(Normal(-2,3),50,70)) ≈ 0.029373438107168350377591231295634273607812172191712
    @test var(truncated(Normal(0,1),-Inf,Inf)) == 1
    @test var(truncated(Normal(0,1),0,+Inf)) ≈ 1 - 2/π
    @test var(truncated(Normal(0,1),-Inf,0)) ≈ 1 - 2/π
    # https://github.com/JuliaStats/Distributions.jl/issues/827
    @test mean(truncated(Normal(1000000,1),0,1000)) ≈ 999.999998998998999001005011019018990904720462367106
    @test var(truncated(Normal(),999000,1e6)) ≥ 0
    @test var(truncated(Normal(1000000,1),0,1000)) ≥ 0
    # https://github.com/JuliaStats/Distributions.jl/issues/624
    @test rand(truncated(Normal(+Inf, 1), 0, 1)) ≈ 1
    @test rand(truncated(Normal(-Inf, 1), 0, 1)) ≈ 0
end
@testset "Truncated normal $trunc" begin
    trunc = truncated(Normal(0, 1), -2, 2)
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
