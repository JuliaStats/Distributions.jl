using Distributions
using Test, Random, ForwardDiff
using ForwardDiff: Dual

@testset "Testing support for continuous distributions" begin
    @testset "Testing $Dist" for
        (Dist, args) in Dict(Arcsine => (0.0, 1.0),
                             Beta => (1.0, 2.0),
                             BetaPrime => (1.0, 1.0),
                             Biweight => (0.0, 1.0),
                             Cauchy => (0.0, 1.0),
                             # Chernoff not parameterised
                             Chi => (2.0,),
                             Chisq => (2.0,),
                             Normal => (0.0, 1.0),
                             LogNormal => (0.0, 1.0),
                             Levy => (0.0, 1.0))
        @testset "Testing $T" for T in [Float64, Float32, Float16, Dual]
            dist = Dist(T.(args)...)
            @test eltype(dist) == typeof(rand(dist)) ==
                eltype(rand(dist, 3)) == typeof(quantile(dist, 0.5))
        end
    end
end
