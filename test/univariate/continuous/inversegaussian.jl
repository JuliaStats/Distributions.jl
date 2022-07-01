@testset "Inner and outer constructors (#1479)" begin
    @test_throws DomainError InverseGaussian(0.0, 0.0)
    @test InverseGaussian(0.0, 0.0; check_args=false) isa InverseGaussian{Float64}
    @test InverseGaussian{Float64}(0.0, 0.0) isa InverseGaussian{Float64}

    @test_throws DomainError Levy(0.0, 0.0)
    @test Levy(0.0, 0.0; check_args=false) isa Levy{Float64}
    @test Levy{Float64}(0.0, 0.0) isa Levy{Float64}
end
