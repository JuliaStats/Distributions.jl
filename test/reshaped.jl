@testset "reshaped.jl" begin
    using Distributions
    using Distributions: ReshapedDistribution
    using Test, Random, LinearAlgebra

    rng = MersenneTwister(1234)

    function test_reshaped(rng, d1, sizes)
        x1 = rand(rng, d1)
        x3 = similar(x1, size(x1)..., 3)
        rand!(rng, d1, x3)
        d1s = map(s -> reshape(d1, s...), sizes)

        # check types
        for (d, s) in zip(d1s, sizes)
            @test d isa ReshapedDistribution{length(s)}
        end

        # incorrect dimensions
        @test_throws ArgumentError reshape(d1, length(d1), 2)
        @test_throws ArgumentError reshape(d1, length(d1), 1, 2)
        @test_throws ArgumentError reshape(d1, -length(d1), -1)

        # size
        for (d, s) in zip(d1s, sizes)
            @test size(d) == s
        end

        # length
        for d in d1s
            @test length(d) == length(d1)
        end

        # rank definition for matrix distributions
        for (d, s) in zip(d1s, sizes)
            if length(s) == 2
                @test rank(d) == minimum(s)
            end
        end

        # support
        for d in d1s, s in sizes
            @test (size(d) == s) ⊻ (length(s) != length(size(d)) || !insupport(d, reshape(x1, s)))
        end

        # mean
        for (d, s) in zip(d1s, sizes)
            @test mean(d) == reshape(mean(d1), s)
        end

        # mode
        for (d, s) in zip(d1s, sizes)
            @test mode(d) == reshape(mode(d1), s)
        end

        # covariance
        for (d, s) in zip(d1s, sizes)
            @test cov(d) == cov(d1)
            if length(s) == 2
                @test cov(d, Val(false)) == reshape(cov(d1), s..., s...)
            end
        end

        # variance
        for (d, s) in zip(d1s, sizes)
            @test var(d) == reshape(var(d1), s)
        end

        # params
        for (d, s) in zip(d1s, sizes)
            @test params(d) == (d1, s)
        end

        # partype
        for d in d1s
            @test partype(d) === partype(d1)
        end

        # eltype
        for d in d1s
            @test eltype(d) === eltype(d1)
        end

        # logpdf
        for (d, s) in zip(d1s, sizes)
            @test logpdf(d, reshape(x1, s)) == logpdf(d1, x1)
            @test logpdf(d, reshape(x3, s..., 3)) == logpdf(d1, x3)
        end

        # loglikelihood
        for (d, s) in zip(d1s, sizes)
            @test loglikelihood(d, reshape(x1, s)) == loglikelihood(d1, x1)
            @test loglikelihood(d, reshape(x3, s..., 3)) == loglikelihood(d1, x3)
        end

        # rand
        for d in d1s
            x = rand(rng, d)
            @test insupport(d, x)
            @test insupport(d1, vec(x))
            @test logpdf(d, x) == logpdf(d1, vec(x))
        end

        # reshape
        for d in d1s
            @test reshape(d, size(d1)...) === d1
            @test reshape(d, size(d1)) === d1
            if d1 isa MultivariateDistribution
                @test vec(d) === d1
            end
        end
        if d1 isa MultivariateDistribution
            @test reshape(d1, size(d1)...) === d1
            @test reshape(d1, size(d1)) === d1
            @test vec(d1) === d1
        end
    end

    @testset "reshape MvNormal" begin
        σ = rand(rng, 16, 16)
        μ = rand(rng, 16)
        d1 = MvNormal(μ, σ * σ')
        sizes = [(4, 4), (8, 2), (2, 8), (1, 16), (16, 1), (4, 2, 2), (2, 4, 2), (2, 2, 2, 2)]
        test_reshaped(rng, d1, sizes)
    end

    @testset "reshape Dirichlet" begin
        α = rand(rng, 36) .+ 1 # mode is only defined if all alpha > 1
        d1 = Dirichlet(α)
        sizes = [
        (6, 6), (4, 9), (9, 4), (3, 12), (12, 3), (1, 36), (36, 1), (6, 3, 2),
        (3, 2, 6), (2, 3, 3, 2),
        ]
        test_reshaped(rng, d1, sizes)
    end

    @testset "special cases" begin
        # MatrixNormal
        rand_posdef_mat(X) = X * X' + I
        U = rand_posdef_mat(rand(5, 5))
        V = rand_posdef_mat(rand(4, 4))
        M = randn(5, 4)
        d = MatrixNormal(M, U, V)

        for v in (vec(d), reshape(d, length(d)), reshape(d, (length(d),)))
            @test v isa MvNormal
            @test mean(v) == vec(M)
            @test cov(v) == kron(V, U)
        end
    end
end
