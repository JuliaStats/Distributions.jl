@testset "DensityInterface" begin
    using DensityInterface

    d_uv_continous = Normal(-1.5, 2.3)
    d_uv_discrete = Poisson(4.7)
    d_mv = MvNormal([2.3 0.4; 0.4 1.2])

    for d in (d_uv_continous, d_uv_discrete, d_mv)
        x = rand(d)
        ref_logd_at_x = logpdf(d, x)
        DensityInterface.test_density_interface(d, x, ref_logd_at_x)

        # Stricter than required by test_density_interface:
        @test logfuncdensity(logdensityof(d)) === d
    end
end
