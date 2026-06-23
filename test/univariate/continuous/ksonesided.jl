using Distributions
using Test

@testset "KSOneSided" begin
    # the sample size `n` is the only parameter, so `partype` is `Int`
    @test @inferred(partype(KSOneSided)) === Int
    @test @inferred(partype(KSOneSided(5))) === Int
end
