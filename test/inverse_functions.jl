VERSION >= v"1.9-" && @testset "InverseFunctions" begin
    using InverseFunctions

    @testset for d in (Normal(1.5, 2.3),)
        # unbounded distribution: can invert cdf at any point in [0..1]
        @testset for f in (cdf, ccdf, logcdf, logccdf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 1.2345)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -Inf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), Inf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -1.2345)
            @test_throws "not defined at 5" inverse(Base.Fix1(f, d))(5)
        end
    end
    @testset for d in (Uniform(1, 2), truncated(Normal(1.5, 2.3), 1, 2))
        # bounded distribution: cannot invert cdf at 0 and 1
        @testset for f in (cdf, ccdf, logcdf, logccdf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 1.2345)
            @test_throws "not defined at 5" inverse(Base.Fix1(f, d))(5)
            @test_throws "not defined at 0" inverse(Base.Fix1(f, d))(0)
            @test_throws "not defined at 1" inverse(Base.Fix1(f, d))(1)
        end
    end

    @testset for d in (Normal(1.5, 2.3), Uniform(1, 2), truncated(Normal(1.5, 2.3), 1, 2))
        # quantile can be inverted everywhere for any continuous distribution
        @testset for f in (quantile, cquantile)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0.1234)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 1)
        end
        @testset for f in (invlogcdf, invlogccdf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -0.1234)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -Inf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0)
        end
    end
end
