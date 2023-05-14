# Tests on Multivariate Normal distributions

import PDMats: ScalMat, PDiagMat, PDMat
if isdefined(PDMats, :PDSparseMat)
    import PDMats: PDSparseMat
end

using Distributions
using LinearAlgebra, Random, Test
using SparseArrays
using FillArrays

###### General Testing

@testset "MvNormal tests" begin
    mu = [1., 2., 3.]
    mu_r = 1.:3.
    va = [1.2, 3.4, 2.6]
    C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]

    h = [1., 2., 3.]
    dv = [1.2, 3.4, 2.6]
    J = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    D = Diagonal(J);
    for (g, μ, Σ) in [
        (@test_deprecated(MvNormal(mu, sqrt(2.0))), mu, Matrix(2.0I, 3, 3)),
        (@test_deprecated(MvNormal(mu_r, sqrt(2.0))), mu_r, Matrix(2.0I, 3, 3)),
        (@test_deprecated(MvNormal(3, sqrt(2.0))), zeros(3), Matrix(2.0I, 3, 3)),
        (@test_deprecated(MvNormal(mu, sqrt.(va))), mu, Matrix(Diagonal(va))),
        (@test_deprecated(MvNormal(mu_r, sqrt.(va))), mu_r, Matrix(Diagonal(va))),
        (@test_deprecated(MvNormal(sqrt.(va))), zeros(3), Matrix(Diagonal(va))),
        (MvNormal(mu, C), mu, C),
        (MvNormal(mu_r, C), mu_r, C),
        (MvNormal(C), zeros(3), C),
        (MvNormal(Symmetric(C)), zeros(3), Matrix(Symmetric(C))),
        (MvNormal(Diagonal(dv)), zeros(3), Matrix(Diagonal(dv))),
        (@test_deprecated(MvNormalCanon(h, 2.0)), h ./ 2.0, Matrix(0.5I, 3, 3)),
        (@test_deprecated(MvNormalCanon(mu_r, 2.0)), mu_r ./ 2.0, Matrix(0.5I, 3, 3)),
        (@test_deprecated(MvNormalCanon(3, 2.0)), zeros(3), Matrix(0.5I, 3, 3)),
        (@test_deprecated(MvNormalCanon(h, dv)), h ./ dv, Matrix(Diagonal(inv.(dv)))),
        (@test_deprecated(MvNormalCanon(mu_r, dv)), mu_r ./ dv, Matrix(Diagonal(inv.(dv)))),
        (@test_deprecated(MvNormalCanon(dv)), zeros(3), Matrix(Diagonal(inv.(dv)))),
        (MvNormalCanon(h, J), J \ h, inv(J)),
        (MvNormalCanon(J), zeros(3), inv(J)),
        (MvNormalCanon(h, D), Diagonal(D) \ h, inv(D)),
        (MvNormalCanon(D), zeros(3), inv(D)),
        (MvNormalCanon(h, Symmetric(D)), D \ h, inv(D)),
        (MvNormalCanon(Hermitian(D)), zeros(3), inv(D)),
        (MvNormal(mu, Symmetric(C)), mu, Matrix(Symmetric(C))),
        (MvNormal(mu_r, Symmetric(C)), mu_r, Matrix(Symmetric(C))),
        (MvNormal(mu, Diagonal(dv)), mu, Matrix(Diagonal(dv))),
        (MvNormal(mu, Symmetric(Diagonal(dv))), mu, Matrix(Diagonal(dv))),
        (MvNormal(mu, Hermitian(Diagonal(dv))), mu, Matrix(Diagonal(dv))),
        (MvNormal(mu_r, Diagonal(dv)), mu_r, Matrix(Diagonal(dv))) ]

        @test mean(g)   ≈ μ
        @test cov(g)    ≈ Σ
        @test invcov(g) ≈ inv(Σ)
        Distributions.TestUtils.test_mvnormal(g, 10^4)

        # conversion between mean form and canonical form
        if isa(g, MvNormal)
            gc = canonform(g)
            @test isa(gc, MvNormalCanon)
            @test length(gc) == length(g)
            @test mean(gc) ≈ mean(g)
            @test cov(gc)  ≈ cov(g)
        else
            @assert isa(g, MvNormalCanon)
            gc = meanform(g)
            @test isa(gc, MvNormal)
            @test length(gc) == length(g)
            @test mean(gc) ≈ mean(g)
            @test cov(gc)  ≈ cov(g)
        end
    end
end


@testset "MvNormal constructor" begin
    mu = [1., 2., 3.]
    C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    J = inv(C)
    h = J \ mu
    @test typeof(MvNormal(mu, PDMat(Array{Float32}(C)))) == typeof(MvNormal(mu, PDMat(C)))
    @test typeof(MvNormal(mu, Array{Float32}(C))) == typeof(MvNormal(mu, PDMat(C)))
    @test typeof(@test_deprecated(MvNormal(mu, 2.0f0))) == typeof(@test_deprecated(MvNormal(mu, 2.0)))

    @test typeof(MvNormalCanon(h, PDMat(Array{Float32}(J)))) == typeof(MvNormalCanon(h, PDMat(J)))
    @test typeof(MvNormalCanon(h, Array{Float32}(J))) == typeof(MvNormalCanon(h, PDMat(J)))
    @test typeof(@test_deprecated(MvNormalCanon(h, 2.0f0))) == typeof(@test_deprecated(MvNormalCanon(h, 2.0)))

    @test typeof(MvNormalCanon(mu, Array{Float16}(h), PDMat(Array{Float32}(J)))) == typeof(MvNormalCanon(mu, h, PDMat(J)))

    d = MvNormal(Array{Float32}(mu), PDMat(Array{Float32}(C)))
    @test convert(MvNormal{Float32}, d) === d
    @test typeof(convert(MvNormal{Float64}, d)) == typeof(MvNormal(mu, PDMat(C)))
    @test typeof(convert(MvNormal{Float64}, d.μ, d.Σ)) == typeof(MvNormal(mu, PDMat(C)))

    d = MvNormalCanon(Array{Float32}(mu), Array{Float32}(h), PDMat(Array{Float32}(J)))
    @test convert(MvNormalCanon{Float32}, d) === d
    @test typeof(convert(MvNormalCanon{Float64}, d)) == typeof(MvNormalCanon(mu, h, PDMat(J)))
    @test typeof(convert(MvNormalCanon{Float64}, d.μ, d.h, d.J)) == typeof(MvNormalCanon(mu, h, PDMat(J)))

    @test MvNormal(mu, I) === @test_deprecated(MvNormal(mu, 1))
    @test MvNormal(mu, 9 * I) === @test_deprecated(MvNormal(mu, 3))
    @test MvNormal(mu, 0.25f0 * I) === @test_deprecated(MvNormal(mu, 0.5))

    @test MvNormal(mu, I) === MvNormal(mu, Diagonal(Ones(length(mu))))
    @test MvNormal(mu, 9 * I) === MvNormal(mu, Diagonal(Fill(9, length(mu))))
    @test MvNormal(mu, 0.25f0 * I) === MvNormal(mu, Diagonal(Fill(0.25f0, length(mu))))

    @test MvNormalCanon(h, I) == MvNormalCanon(h, Diagonal(Ones(length(h))))
    @test MvNormalCanon(h, 9 * I) == MvNormalCanon(h, Diagonal(Fill(9, length(h))))
    @test MvNormalCanon(h, 0.25f0 * I) == MvNormalCanon(h, Diagonal(Fill(0.25f0, length(h))))

    @test typeof(MvNormalCanon(h, I)) === typeof(MvNormalCanon(h, Diagonal(Ones(length(h)))))
    @test typeof(MvNormalCanon(h, 9 * I)) === typeof(MvNormalCanon(h, Diagonal(Fill(9, length(h)))))
    @test typeof(MvNormalCanon(h, 0.25f0 * I)) === typeof(MvNormalCanon(h, Diagonal(Fill(0.25f0, length(h)))))
end

@testset "MvNormal 32-bit logpdf" begin
    # Test 32-bit logpdf
    mu = [1., 2., 3.]
    C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    d = MvNormal(mu, PDMat(C))
    X = [1., 2., 3.]

    d32 = convert(MvNormal{Float32}, d)
    X32 = convert(AbstractArray{Float32}, X)

    @test isa(logpdf(d32, X32), Float32)
    @test logpdf(d32, X32) ≈ logpdf(d, X)
end

##### Random sampling from MvNormalCanon with sparse precision matrix
if isdefined(PDMats, :PDSparseMat)
    @testset "Sparse MvNormalCanon random sampling" begin
        n = 20
        nsamp = 100_000
        Random.seed!(1234)

        J = sprandn(n, n, 0.25)
        J = J'J + I
        Σ = inv(Matrix(J))
        J = PDSparseMat(J)
        μ = zeros(n)

        d_prec_sparse = MvNormalCanon(μ, J*μ, J)
        d_prec_dense = MvNormalCanon(μ, J*μ, PDMat(Matrix(J)))
        d_cov_dense = MvNormal(μ, PDMat(Symmetric(Σ)))

        x_prec_sparse = rand(d_prec_sparse, nsamp)
        x_prec_dense = rand(d_prec_dense, nsamp)
        x_cov_dense = rand(d_cov_dense, nsamp)

        dists = [d_prec_sparse, d_prec_dense, d_cov_dense]
        samples = [x_prec_sparse, x_prec_dense, x_cov_dense]
        tol = 1e-16
        se = sqrt.(diag(Σ) ./ nsamp)
        #=
        The cholesky decomposition of sparse matrices is performed by `SuiteSparse.CHOLMOD`,
        which returns a different decomposition than the `Base.LinearAlgebra` function (which uses
        LAPACK). These different Cholesky routines produce different factorizations (since the
        Cholesky factorization is not in general unique).  As a result, the random samples from
        an `MvNormalCanon` distribution with a sparse precision matrix are not in general
        identical to those from an `MvNormalCanon` or `MvNormal`, even if the seeds are
        identical.  As a result, these tests only check for approximate statistical equality,
        rather than strict numerical equality of the samples.
        =#
        for i in 1:3, j in 1:3
            @test all(abs.(mean(samples[i]) .- μ) .< 2se)
            loglik_ii = [logpdf(dists[i], samples[i][:, k]) for k in 1:100_000]
            loglik_ji = [logpdf(dists[j], samples[i][:, k]) for k in 1:100_000]
            # test average likelihood ratio between distribution i and sample j are small
            @test mean((loglik_ii .- loglik_ji).^2) < tol
        end
    end
end


##### MLE

# a slow but safe way to implement MLE for verification

function _gauss_mle(x::AbstractMatrix{<:Real})
    mu = vec(mean(x, dims=2))
    z = x .- mu
    C = (z * z') * (1/size(x,2))
    return mu, C
end

function _gauss_mle(x::AbstractMatrix{<:Real}, w::AbstractVector{<:Real})
    sw = sum(w)
    mu = (x * w) * (1/sw)
    z = x .- mu
    C = (z * (Diagonal(w) * z')) * (1/sw)
    LinearAlgebra.copytri!(C, 'U')
    return mu, C
end

@testset "MvNormal MLE" begin
    x = randn(3, 200) .+ randn(3) * 2.
    w = rand(200)
    u, C = _gauss_mle(x)
    uw, Cw = _gauss_mle(x, w)

    g = fit_mle(MvNormal, suffstats(MvNormal, x))
    @test isa(g, FullNormal)
    @test mean(g) ≈ u
    @test cov(g)  ≈ C

    g = fit_mle(MvNormal, x)
    @test isa(g, FullNormal)
    @test mean(g) ≈ u
    @test cov(g)  ≈ C

    g = fit_mle(MvNormal, x, w)
    @test isa(g, FullNormal)
    @test mean(g) ≈ uw
    @test cov(g)  ≈ Cw

    g = fit_mle(IsoNormal, x)
    @test isa(g, IsoNormal)
    @test g.μ       ≈ u
    @test g.Σ.value ≈ mean(diag(C))

    g = fit_mle(IsoNormal, x, w)
    @test isa(g, IsoNormal)
    @test g.μ       ≈ uw
    @test g.Σ.value ≈ mean(diag(Cw))

    g = fit_mle(DiagNormal, x)
    @test isa(g, DiagNormal)
    @test g.μ      ≈ u
    @test g.Σ.diag ≈ diag(C)

    g = fit_mle(DiagNormal, x, w)
    @test isa(g, DiagNormal)
    @test g.μ      ≈ uw
    @test g.Σ.diag ≈ diag(Cw)
end

@testset "MvNormal affine transformations" begin
    @testset "moment identities" begin
        for n in 1:5                       # dimension
            # distribution
            μ = randn(n)
            for Σ in (randn(n, n) |> A -> A*A',  # dense
                      Diagonal(abs2.(randn(n))), # diagonal
                      abs2(randn()) * I)         # scaled unit
                d = MvNormal(μ, Σ)

                # random arrays for transformations
                c = randn(n)
                m = rand(1:n)
                B = randn(m, n)
                b = randn(n)

                d_c = d + c
                c_d = c + d
                @test mean(d_c) == mean(c_d) == μ + c
                @test cov(c_d) == cov(d_c) == cov(d)

                d_c = d - c
                @test mean(d_c) == μ - c
                @test cov(d_c) == cov(d)

                B_d = B * d
                @test B_d isa MvNormal
                @test length(B_d) == m
                @test mean(B_d) == B * μ
                @test cov(B_d) ≈ B * Σ * B'

                b_d = dot(b, d)
                d_b = dot(b, d)
                @test b_d isa Normal && d_b isa Normal
                @test mean(b_d) ≈ mean(d_b) ≈ dot(b, μ)
                @test var(b_d) ≈ var(d_b) ≈ dot(b, Σ * b)
            end
        end
    end

    @testset "dimension mismatch errors" begin
        d4 = MvNormal(zeros(4), Diagonal(ones(4)))
        o3 = ones(3)
        @test_throws DimensionMismatch d4 + o3
        @test_throws DimensionMismatch o3 + d4
        @test_throws DimensionMismatch ones(3, 3) * d4
        @test_throws DimensionMismatch dot(o3, d4)
    end
end

@testset "MvNormal: Sampling with integer-valued parameters (#1004)" begin
    d = MvNormal([0, 0], Diagonal([1, 1]))
    @test rand(d) isa Vector{Float64}
    @test rand(d, 10) isa Matrix{Float64}
    @test rand(d, (3, 2)) isa Matrix{Vector{Float64}}

    # evaluation of `logpdf`
    # (bug fixed by https://github.com/JuliaStats/Distributions.jl/pull/1429)
    x = rand(d)
    @test logpdf(d, x) ≈ logpdf(Normal(), x[1]) + logpdf(Normal(), x[2])
end
