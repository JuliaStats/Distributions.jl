@testset "DensityInterface" begin
    using DensityInterface

    d_uv_continous = Normal(-1.5, 2.3)
    d_uv_discrete = Poisson(4.7)
    d_mv = MvNormal([2.3 0.4; 0.4 1.2])
    d_av = Distributions.MatrixReshaped(MvNormal(rand(10)), 2, 5)

    for d in (d_uv_continous, d_uv_discrete, d_mv, d_av)
        x = rand(d)
        ref_logd_at_x = logpdf(d, x)
        DensityInterface.test_density_interface(d, x, ref_logd_at_x)

        # Stricter than required by test_density_interface:
        @test logfuncdensity(logdensityof(d)) === d
    end

    @test_throws ArgumentError logdensityof(d_uv_continous, [rand(d_uv_continous) for i in 1:3])
    @test_throws ArgumentError logdensityof(d_mv, hcat([rand(d_mv) for i in 1:3]...))
    @test_throws ArgumentError logdensityof(d_av, [rand(d_av) for i in 1:3])
end
