using Distributions
using Test

@testset "Skellam mgf near t = 0" begin
    # `exp(t) - 1` cancels catastrophically for small `t`, which large rates amplify
    d = Skellam(1e10, 2e10)
    t = 1e-12
    ref = exp(big(1e10) * expm1(big(t)) + big(2e10) * expm1(-big(t)))
    @test mgf(d, t) ≈ ref
end
