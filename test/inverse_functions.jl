@testset "InverseFunctions" begin
    using InverseFunctions

    @testset for d in (Normal(1.5, 2.3), Uniform(1, 2), truncated(Normal(1.5, 2.3), 1, 2))
        @testset for f in (cdf, ccdf, logcdf, logccdf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 1.2345)
        end
        @testset for f in (quantile, cquantile)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0.1234)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 1)
        end
        @testset for f in (invlogcdf, invlogccdf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -Inf)
            InverseFunctions.test_inverse(Base.Fix1(f, d), -0.1234)
            InverseFunctions.test_inverse(Base.Fix1(f, d), 0)
        end
    end
end
