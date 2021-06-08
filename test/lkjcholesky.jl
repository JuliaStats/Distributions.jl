using Distributions
using Random
using LinearAlgebra
using Test

function test_draw(d::LKJCholesky, x; check_uplo=true)
    @test insupport(d, x)
    check_uplo && @test x.uplo == d.uplo
end
function test_draws(d::LKJCholesky, xs; check_uplo=true)
    @test all(x -> insupport(d, x), xs)
    check_uplo && @test all(x -> x.uplo == d.uplo, xs)
    @testset "LKJCholesky marginals" begin
        ndraws = length(xs)
        p = dim(d)    
        dmat = LKJ(p, d.η)
        marginal = Distributions._marginal(dmat)    
        α = 0.05
        L = sum(1:(p - 1))
        zs = Array{eltype(d)}(undef, p, p, ndraws)
        for k in 1:ndraws
            zs[:, :, k] = Matrix(xs[k])
        end
        for i in 1:p, j in 1:(i-1)
            @test pvalue_kolmogorovsmirnoff(zs[i, j, :], marginal) >= α / L
        end
    end
end

@testset "LKJCholesky" begin
    @testset "Constructors" begin
        @testset for p in (4, 5), η in (2, 3.5)
            d = LKJCholesky(p, η)
            @test d.d == p
            @test d.η == η
            @test d.uplo == 'L'
            @test d.logc0 == LKJ(p, η).logc0
        end

        @test_throws ArgumentError LKJCholesky(0, 2)
        @test_throws ArgumentError LKJCholesky(4, 0.0)
        @test_throws ArgumentError LKJCholesky(4, -1)

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
        @testset for p in (4, 5), η in (2, 3.5), uplo in ('L', 'U')
            d = LKJCholesky(p, η, uplo)
            @test dim(d) == p
            @test size(d) == (p, p)
            @test Distributions.params(d) == (d.d, d.η, d.uplo)
            @test partype(d) <: Float64

            m = mode(d)
            @test m isa Cholesky{eltype(d)}
            @test Matrix(m) ≈ I
        end
        @test_broken partype(LKJCholesky(2, 4f0)) <: Float32

        @testset "insupport" begin
            @test insupport(LKJCholesky(40, 2, 'U'), cholesky(rand(LKJ(40, 2))))
            @test insupport(LKJCholesky(40, 2), cholesky(rand(LKJ(40, 2))))
            @test !insupport(LKJCholesky(40, 2), cholesky(rand(LKJ(41, 2))))
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
            logdetJ = sum(i -> (i - p) * log(x.UL[i, i]), 1:p)

            @test logpdf(dchol, x) ≈ logpdf(d, z) - logdetJ
            @test logpdf(dchol, x_L) ≈ logpdf(dchol, x)

            @test pdf(dchol, x) ≈ exp(logpdf(dchol, x))
            @test pdf(dchol, x_L) ≈ pdf(dchol, x)

            @test loglikelihood(dchol, x) ≈ logpdf(dchol, x)
            xs = cholesky.(rand(d, 10))
            @test loglikelihood(dchol, xs) ≈ sum(logpdf(dchol, x) for x in xs)
        end
    end

    @testset "Sampling" begin
        @testset "rand" begin
            @testset for p in (2, 4, 10), η in (0.5, 1, 3), uplo in ('L', 'U')
                d = LKJCholesky(p, η, uplo)
                test_draw(d, rand(d))
                test_draws(d, rand(d, 10^4))
            end
            @test_broken rand(LKJCholesky(5, Inf)) ≈ I
        end

        @testset "rand!" begin
            @testset for p in (2, 4, 10), η in (0.5, 1, 3), uplo in ('L', 'U')
                d = LKJCholesky(p, η, uplo)
                x = Cholesky(Matrix{Float64}(undef, p, p), uplo, 0)
                rand!(d, x)
                test_draw(d, x)
                # test that uplo of Cholesky object is respected
                x2 = Cholesky(Matrix{Float64}(undef, p, p), uplo == 'L' ? 'U' : 'L', 0)
                rand!(d, x2)
                test_draw(d, x2; check_uplo = false)

                # allocating
                xs = Vector{typeof(x)}(undef, 10^4)
                rand!(d, xs)
                test_draws(d, xs)

                F2 = cholesky(exp(Symmetric(randn(p, p))))
                xs2 = [deepcopy(F2) for _ in 1:10^4]
                xs2[1] = cholesky(exp(Symmetric(randn(p + 1, p + 1))))
                rand!(d, xs2)
                test_draws(d, xs2)

                # non-allocating
                F3 = cholesky(exp(Symmetric(randn(p, p))))
                xs3 = [deepcopy(F3) for _ in 1:10^4]
                rand!(d, xs3)
                test_draws(d, xs3; check_uplo = uplo == 'U')
            end
        end
    end
end