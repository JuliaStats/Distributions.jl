using Distributions, Test

@testset "skewness identities" begin
    @test skewness(Kumaraswamy(1, 1)) ≈ 0 atol=1e-12
    # K(a, 1) = Beta(a, 1) and K(1, b) = Beta(1, b). Beta uses
    # its independent closed-form skewness, rather than raw moments.
    for shape in (0.2, 0.5, 0.99, 1.01, 2.0, 5.0)
        @test skewness(Kumaraswamy(shape, 1.0)) ≈ skewness(Beta(shape, 1.0)) rtol=1e-10 atol=1e-12
        @test skewness(Kumaraswamy(1.0, shape)) ≈ skewness(Beta(1.0, shape)) rtol=1e-10 atol=1e-12
    end
    # Density 4x(1-x²): raw moments 8/15, 1/3, 8/35 give this
    # exact skewness without calling the distribution's moment helper.
    @test skewness(Kumaraswamy(2, 2)) ≈ -32 / (77sqrt(11)) rtol=1e-10 atol=1e-12
end
