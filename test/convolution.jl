@testset "discrete univariate" begin

    @testset "Bernoulli" begin
        d1 = Bernoulli(0.1)
        d2 = convolve(d1, d1)

        @test isa(d2, Binomial)
        @test d2.n == 2
        @test d2.p == 0.1

        # cannot convolve a Binomial with a Bernoulli
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 == p2
        d3 = Bernoulli(0.2)
        @test_throws ArgumentError convolve(d1, d3)

    end

    @testset "Binomial" begin
        d1 = Binomial(2, 0.1)
        d2 = Binomial(5, 0.1)
        d3 = convolve(d1, d2)

        @test isa(d3, Binomial)
        @test d3.n == 7
        @test d3.p == 0.1

        # only works if p1 == p2
        d4 = Binomial(2, 0.2)
        @test_throws ArgumentError convolve(d1, d4)

    end

    @testset "NegativeBinomial" begin
        d1 = NegativeBinomial(4, 0.1)
        d2 = NegativeBinomial(1, 0.1)
        d3 = convolve(d1, d2)

        isa(d3, NegativeBinomial)
        @test d3.r == 5
        @test d3.p == 0.1

        d4 = NegativeBinomial(1, 0.2)
        @test_throws ArgumentError convolve(d1, d4)
    end


    @testset "Geometric" begin
        d1 = Geometric(0.2)
        d2 = convolve(d1, d1)

        @test isa(d2, NegativeBinomial)
        @test d2.p == 0.2

        # cannot convolve a Geometric with a NegativeBinomial
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 == p2
        d3 = Geometric(0.5)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Poisson" begin
        d1 = Poisson(0.1)
        d2 = Poisson(0.4)
        d3 = convolve(d1, d2)

        @test isa(d3, Poisson)
        @test d3.λ == 0.5
    end

end

@testset "continuous univariate" begin

    @testset "Gaussian" begin
        d1 = Normal(0.1, 0.2)
        d2 = Normal(0.25, 1.7)
        d3 = convolve(d1, d2)

        @test isa(d3, Normal)
        @test d3.μ == 0.35
        @test d3.σ == hypot(0.2, 1.7)
    end

    @testset "Cauchy" begin
        d1 = Cauchy(0.2, 0.7)
        d2 = Cauchy(1.9, 0.8)
        d3 = convolve(d1, d2)

        @test isa(d3, Cauchy)
        @test d3.μ == 2.1
        @test d3.σ == 1.5
    end

    @testset "Chisq" begin
        d1 = Chisq(0.1)
        d2 = Chisq(0.3)
        d3 = convolve(d1, d2)

        @test isa(d3, Chisq)
        @test d3.ν == 0.4
    end

    @testset "Exponential" begin
        d1 = Exponential(0.7)
        d2 = convolve(d1, d1)

        @test isa(d2, Gamma)
        @test d2.α == 2
        @test d2.θ == 0.7

        # cannot convolve an Exponential with a Gamma
        @test_throws MethodError convolve(d1, d2)

        # only works if θ1 == θ2
        d3 = Exponential(0.2)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Gamma" begin
        d1 = Gamma(0.1, 1.7)
        d2 = Gamma(0.5, 1.7)
        d3 = convolve(d1, d2)

        @test isa(d3, Gamma)
        @test d3.α == 0.6
        @test d3.θ == 1.7

        # only works if θ1 == θ4
        d4 = Gamma(1.2, 0.4)
        @test_throws ArgumentError convolve(d1, d4)
    end

end

@testset "continuous multivariate" begin

    @testset "iso-normal" begin

        in1 = MvNormal([1.2, 0.3], 2)
        in2 = MvNormal([-2.0, 6.9], 0.5)

        zm1 = MvNormal(2, 1.9)
        zm2 = MvNormal(2, 5.2)

        for (d1, d2) in ((in1, in2), (zm1, zm2), (in1, zm1))
            d3 = convolve(d1, d2)
            expected_Σ = dot(d1.Σ.value + d2.Σ.value, d1.Σ.value + d2.Σ.value)

            @test isa(d3, IsoNormal)
            @test d3.μ == d1.μ .+ d2.μ
            @test d3.Σ == ScalMat(2, expected_Σ)
        end

        # erroring
        in3 = MvNormal([1, 2, 3], 0.2)
        @test_throws ArgumentError convolve(in1, in3)
    end

    @testset "diag-normal" begin

        dn1 = MvNormal([0.0, 4.7], [0.1, 1.8])
        dn2 = MvNormal([-3.4, 1.2], [3.2, 0.2])

        zm1 = MvNormal([1.2, 0.3])
        zm2 = MvNormal([-0.8, 1.0])


        for (d1, d2) in ((dn1, dn2), (zm1, zm2), (dn1, zm1))
            d3 = convolve(d1, d2)
            expected_Σ = dot.(d1.Σ.diag + d2.Σ.diag, d1.Σ.diag + d2.Σ.diag)

            @test isa(d3, DiagNormal)
            @test d3.μ == d1.μ .+ d2.μ
            @test d3.Σ.diag == expected_Σ  # == not defined for PDiagMat
        end

        # erroring
        dn3 =  MvNormal([1, 2, 3], [4.2, 5.3, 2.1])
        @test_throws ArgumentError convolve(dn1, dn3)
    end

    @testset "full-normal" begin

        L1 = [1 0; 2 1]
        fn1 = MvNormal(ones(2), cov(L1 * L1'))
        L2 = [1.2 0; 3.4 6.6]
        fn2 = MvNormal([2.1, 0.4], cov(L2 * L2'))
        L3 = [2.1 0; 0.3 1.2]
        zm1 = MvNormal(cov(L3 * L3'))
        L4 = [4.1 0; 1.7 6.4]
        zm2 = MvNormal(cov(L4 * L4'))

        for (d1, d2) in ((fn1, fn2), (zm1, zm2), (fn1, zm1))
            d3 = convolve(d1, d2)
            expected_Σ = d1.Σ.mat + d2.Σ.mat

            @test isa(d3, FullNormal)
            @test d3.μ == d1.μ .+ d2.μ
            @test d3.Σ.mat == expected_Σ  # == not defined for PDMat
        end

        # erroring
        L5 = [4.1 0 0; 1.7 6.4 0; 2.1 8.5 0]
        fn3 =  MvNormal(zeros(3), cov(L5 * L5'))
        @test_throws ArgumentError convolve(fn1, fn3)
    end

    @testset "mixed" begin
        iso = MvNormal([1.2, 0.3], 2)
        diag = MvNormal([0.0, 4.7], [0.1, 1.8])
        L1 = [1 0; 2 1]
        full = MvNormal(ones(2), cov(L1 * L1'))

        for (d1, d2) in ((iso, diag), (diag, full), (full, iso))
            d3 = convolve(d1, d2)
            expected_Σ = Matrix(d1.Σ) + Matrix(d2.Σ)

            @test isa(d3, MvNormal)
            @test d3.μ == d1.μ .+ d2.μ
            @test d3.Σ.mat == expected_Σ  # == not defined for PDMat
        end
    end
end
