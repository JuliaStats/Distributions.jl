using Distributions
using Test

@testset "Hypergeometric" begin
    # the parameters are always stored as `Int`, so `partype` is `Int`
    @test @inferred(partype(Hypergeometric)) === Int
    @test @inferred(partype(Hypergeometric(2, 2, 2))) === Int
    @test @inferred(partype(Hypergeometric(2.0, 2, 2))) === Int
end
