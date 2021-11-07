@testset "DensityInterface" begin
    using DensityInterface

    d_uv_continous = Normal(-1.5, 2.3)
    d_uv_discrete = Poisson(4.7)
    d_mv = MvNormal([2.3 0.4; 0.4 1.2])
    d_av = Distributions.MatrixReshaped(MvNormal(rand(10)), 2, 5)

    @testset "Distribution" begin
        for d in (d_uv_continous, d_uv_discrete, d_mv, d_av)
            x = rand(d)
            ref_logd_at_x = logpdf(d, x)
            DensityInterface.test_density_interface(d, x, ref_logd_at_x)

            # Stricter than required by test_density_interface:
            @test logfuncdensity(logdensityof(d)) === d
        end

        for di_func in (logdensityof, densityof)
            @test_throws ArgumentError di_func(d_uv_continous, [rand(d_uv_continous) for i in 1:3])
            @test_throws ArgumentError di_func(d_mv, hcat([rand(d_mv) for i in 1:3]...))
            @test_throws ArgumentError di_func(d_av, [rand(d_av) for i in 1:3])
        end
    end

    @testset "IIDDensity" begin
        x = [rand(d_mv) for i in 1:3]
        ref_logd_at_x = loglikelihood(d_mv, x)
        d = IIDDensity(d_mv)

        DensityInterface.test_density_interface(d, x, ref_logd_at_x)
        DensityInterface.test_density_interface(d, hcat(x...), ref_logd_at_x)

        # Stricter than required by test_density_interface:
        @test logfuncdensity(logdensityof(d)) === d
    end
end
