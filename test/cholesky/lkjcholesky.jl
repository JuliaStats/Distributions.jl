using Distributions
using Random
using LinearAlgebra
using Test
using FiniteDifferences

@testset "LKJCholesky" begin
    function test_draw(d::LKJCholesky, x; check_uplo=true)
        @test insupport(d, x)
        check_uplo && @test x.uplo == d.uplo
    end
    function test_draws(d::LKJCholesky, xs; check_uplo=true, nkstests=1)
        @test all(x -> insupport(d, x), xs)
        check_uplo && @test all(x -> x.uplo == d.uplo, xs)

        p = d.d
        dmat = LKJ(p, d.η)
        marginal = Distributions._marginal(dmat)    
        ndraws = length(xs)
        zs = Array{eltype(d)}(undef, p, p, ndraws)
        for k in 1:ndraws
            zs[:, :, k] = Matrix(xs[k])
        end

        @testset "LKJCholesky marginal moments" begin
            @test mean(zs; dims=3)[:, :, 1] ≈ I atol=0.1
            @test var(zs; dims=3)[:, :, 1] ≈ var(marginal) * (ones(p, p) - I) atol=0.1
            @testset for n in 2:5
                for i in 1:p, j in 1:(i-1)
                    @test moment(zs[i, j, :], n) ≈ moment(rand(marginal, ndraws), n) atol=0.1
                end
            end
        end

        @testset "LKJCholesky marginal KS test" begin
            α = 0.01
            L = sum(1:(p - 1))
            for i in 1:p, j in 1:(i-1)
                @test pvalue_kolmogorovsmirnoff(zs[i, j, :], marginal) >= α / L / nkstests
            end
        end
    end

    # Compute logdetjac of ϕ: L → L L' where only strict lower triangle of L and L L' are unique
    function cholesky_inverse_logdetjac(L)
        size(L, 1) == 1 && return 0.0
        J = jacobian(central_fdm(5, 1), cholesky_vec_to_corr_vec, stricttril_to_vec(L))[1]
        return logabsdet(J)[1]
    end
    stricttril_to_vec(L) = [L[i, j] for i in axes(L, 1) for j in 1:(i - 1)]
    function vec_to_stricttril(l)
        n = length(l)
        p = Int((1 + sqrt(8n + 1)) / 2)
        L = similar(l, p, p)
        fill!(L, 0)
        k = 1
        for i in 1:p, j in 1:(i - 1)
            L[i, j] = l[k]
            k += 1
        end
        return L
    end
    function cholesky_vec_to_corr_vec(l)
        L = vec_to_stricttril(l)
        for i in axes(L, 1)
            w = view(L, i, 1:(i-1))
            wnorm = norm(w)
            if wnorm > 1
                w ./= wnorm
                wnorm = 1
            end
            L[i, i] = sqrt(1 - wnorm^2)
        end
        return stricttril_to_vec(L * L')
    end

    @testset "Constructors" begin
        @testset for p in (4, 5), η in (2, 3.5)
            d = LKJCholesky(p, η)
            @test d.d == p
            @test d.η == η
            @test d.uplo == 'L'
            @test d.logc0 == LKJ(p, η).logc0
        end

        @test_throws DomainError LKJCholesky(0, 2)
        @test_throws DomainError LKJCholesky(4, 0.0)
        @test_throws DomainError LKJCholesky(4, -1)

        for uplo in (:U, 'U')
            d = LKJCholesky(4, 2, uplo)
            @test d.uplo === 'U'
        end
        for uplo in (:L, 'L')
            d = LKJCholesky(4, 2, uplo)
            @test d.uplo === 'L'
        end
        @test_throws ArgumentError LKJCholesky(4, 2, :F)
        @test_throws ArgumentError LKJCholesky(4, 2, 'F')
    end

    @testset "REPL display" begin
        d = LKJCholesky(5, 1)
        @test sprint(show, d) == "$(typeof(d))(\nd: 5\nη: 1.0\nuplo: L\n)\n"
    end

    @testset "Conversion" begin
        d = LKJCholesky(5, 3.5)
        @test convert(LKJCholesky{Float64}, d) === d

        df0_1 = convert(LKJCholesky{Float32}, d)
        @test df0_1 isa LKJCholesky{Float32}
        @test df0_1.d == d.d
        @test df0_1.η ≈ d.η
        @test df0_1.uplo == d.uplo
        @test df0_1.logc0 ≈ d.logc0

        df0_2 = convert(LKJCholesky{BigFloat}, d.d, d.η, d.uplo, d.logc0)
        @test df0_2 isa LKJCholesky{BigFloat}
        @test df0_2.d == d.d
        @test df0_2.η ≈ d.η
        @test df0_2.uplo == d.uplo
        @test df0_2.logc0 ≈ d.logc0
    end

    @testset "properties" begin
        @testset for p in (4, 5), η in (0.5, 1, 2, 3.5), uplo in ('L', 'U')
            d = LKJCholesky(p, η, uplo)
            @test d.d == p
            @test size(d) == (p, p)
            @test Distributions.params(d) == (d.d, d.η, d.uplo)
            @test partype(d) <: Float64

            if η > 1
                m = mode(d)
                @test m isa Cholesky{eltype(d)}
                @test Matrix(m) ≈ I
            else
                @test_throws DomainError(η, "LKJCholesky: mode is defined only when η > 1.") mode(d)
            end
            m = mode(d; check_args = false)
            @test m isa Cholesky{eltype(d)}
            @test Matrix(m) ≈ I
        end
        for (d, η) in ((2, 4), (2, 1), (3, 1)), T in (Float32, Float64)
            @test @inferred(partype(LKJCholesky(d, T(η)))) === T
        end

        @testset "insupport" begin
            @test insupport(LKJCholesky(40, 2, 'U'), cholesky(rand(LKJ(40, 2))))
            @test insupport(LKJCholesky(40, 2), cholesky(rand(LKJ(40, 2))))
            @test !insupport(LKJCholesky(40, 2), cholesky(rand(LKJ(41, 2))))
           for (d, η) in ((2, 4), (2, 1), (3, 1)), T in (Float32, Float64)
                @test @inferred(logpdf(LKJCholesky(40, T(2)), cholesky(T.(rand(LKJ(41, 2)))))) === T(-Inf)
            end
            z = rand(LKJ(40, 1))
            z .+= exp(Symmetric(randn(size(z)))) .* 1e-8
            x = cholesky(z)
            @test !insupport(LKJCholesky(4, 2), x)
        end
    end

    @testset "Evaluation" begin
        @testset for p in (1, 4, 10), η in (0.5, 1, 3)
            d = LKJ(p, η)
            dchol = LKJCholesky(p, η)
            z = rand(d)
            x = cholesky(z)
            x_L = typeof(x)(Matrix(x.L), 'L', x.info)
            logdetJ = sum(i -> (p - i) * log(x.UL[i, i]), 1:p)
            logdetJ_approx = cholesky_inverse_logdetjac(x.L)
            @test logdetJ ≈ logdetJ_approx

            @test logpdf(dchol, x) ≈ logpdf(d, z) + logdetJ
            @test logpdf(dchol, x_L) ≈ logpdf(dchol, x)

            @test pdf(dchol, x) ≈ exp(logpdf(dchol, x))
            @test pdf(dchol, x_L) ≈ pdf(dchol, x)

            @test loglikelihood(dchol, x) ≈ logpdf(dchol, x)
            xs = cholesky.(rand(d, 10))
            @test loglikelihood(dchol, xs) ≈ sum(logpdf(dchol, x) for x in xs)
        end
    end

    @testset "Sampling" begin
        rng = MersenneTwister(66)
        nkstests = 4 # use for appropriate Bonferroni correction for KS test

        @testset "rand" begin
            @test rand(LKJCholesky(1, 0.5)).factors == ones(1, 1)
            @testset for p in (2, 4, 10), η in (0.5, 1, 3), uplo in ('L', 'U')
                d = LKJCholesky(p, η, uplo)
                test_draw(d, rand(rng, d))
                test_draws(d, rand(rng, d, 10^4); nkstests=nkstests)
            end
            @test_broken rand(rng, LKJCholesky(5, Inf)) ≈ I
        end

        @testset "rand!" begin
            @testset for p in (2, 4, 10), η in (0.5, 1, 3), uplo in ('L', 'U')
                d = LKJCholesky(p, η, uplo)
                x = Cholesky(Matrix{Float64}(undef, p, p), uplo, 0)
                rand!(rng, d, x)
                test_draw(d, x)
                x = Cholesky(Matrix{Float64}(undef, p, p), uplo, 0)
                rand!(d, x)
                test_draw(d, x)

                # test that uplo of Cholesky object is respected
                x2 = Cholesky(Matrix{Float64}(undef, p, p), uplo == 'L' ? 'U' : 'L', 0)
                rand!(rng, d, x2)
                test_draw(d, x2; check_uplo = false)

                # allocating
                xs = Vector{typeof(x)}(undef, 10^4)
                rand!(rng, d, xs)
                test_draws(d, xs; nkstests=nkstests)

                F2 = cholesky(exp(Symmetric(randn(rng, p, p))))
                xs2 = [deepcopy(F2) for _ in 1:10^4]
                xs2[1] = cholesky(exp(Symmetric(randn(rng, p + 1, p + 1))))
                rand!(rng, d, xs2)
                test_draws(d, xs2; nkstests=nkstests)

                # non-allocating
                F3 = cholesky(exp(Symmetric(randn(rng, p, p))))
                xs3 = [deepcopy(F3) for _ in 1:10^4]
                rand!(rng, d, xs3)
                test_draws(d, xs3; check_uplo = uplo == 'U', nkstests=nkstests)
            end
        end
    end
end
