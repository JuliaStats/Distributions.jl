using Distributions
using Test

@testset "KSDist" begin
    # the sample size `n` is the only parameter, so `partype` is `Int`
    @test @inferred(partype(KSDist)) === Int
    @test @inferred(partype(KSDist(5))) === Int
end
