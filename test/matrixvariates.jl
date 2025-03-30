using Distributions
using Random
using LinearAlgebra
using PDMats
using Statistics
using Test
import JSON
import Distributions: _univariate, _multivariate, _rand_params

@testset "matrixvariates" begin
#=
    1. baseline tests
    2. compare 1 x 1 matrix-variate with univariate
    3. compare row/column matrix-variate with multivariate
    4. compare density evaluation against archived Stan output
    5. special, distribution-specific tests
    6. main testing method
    7. run matrix-variate unit tests
=#

#  =============================================================================
#  1. baseline test
#  =============================================================================

#  --------------------------------------------------
#  Check that a random draw from d has the right properties
#  --------------------------------------------------

function test_draw(d::MatrixDistribution, X::AbstractMatrix)
    @test size(d) == size(X)
    @test size(d, 1) == size(X, 1)
    @test size(d, 2) == size(X, 2)
    @test length(d) == length(X)
    @test rank(d) == rank(X)
    @test insupport(d, X)
    @test logpdf(d, X) ≈ log(pdf(d, X))
    @test logpdf(d, [X, X]) ≈ log.(pdf(d, [X, X]))
    @test loglikelihood(d, X) ≈ logpdf(d, X)
    @test loglikelihood(d, [X, X]) ≈ 2 * logpdf(d, X)
    if d isa MatrixFDist
        # Broken since `pdadd` is not defined for SubArray
        @test_broken loglikelihood(d, cat(X, X; dims=3)) ≈ 2 * logpdf(d, X)
    else
        @test loglikelihood(d, cat(X, X; dims=3)) ≈ 2 * logpdf(d, X)
    end
    nothing
end

test_draw(d::MatrixDistribution) = test_draw(d, rand(d))

#  --------------------------------------------------
#  Check that sample quantities are close to population quantities
#  --------------------------------------------------

function test_draws(d::MatrixDistribution, draws::AbstractArray{<:AbstractMatrix})
    @test mean(draws) ≈ mean(d) rtol = 0.01
    draws_matrix = mapreduce(vec, hcat, draws)
    @test cov(draws_matrix; dims=2) ≈ cov(d) rtol = 0.1
    nothing
end

function test_draws(d::LKJ, draws::AbstractArray{<:AbstractMatrix})
    @test isapprox(mean(draws), mean(d), atol = 0.1)
    @test isapprox(var(draws), var(d) , atol = 0.1)
    nothing
end

function test_draws(d::MatrixDistribution, M::Integer)
    rng = MersenneTwister(123)
    @testset "Testing matrix-variates with $key" for (key, func) in
        Dict("rand(...)" => [rand, rand],
             "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])
        test_draws(d, func[2](d, M))
    end
    @testset "Testing matrix-variates with $key" for (key, func) in
        Dict("rand!(..., false)" => [(dist, X) -> rand!(dist, X, false), rand!],
             "rand!(rng, ..., false)" => [(dist, X) -> rand!(rng, dist, X, false), (dist, X) -> rand!(rng, dist, X)])
        m = [Matrix{partype(d)}(undef, size(d)) for _ in Base.OneTo(M)]
        x = func[1](d, m)
        @test x ≡ m
        @test isapprox(mean(x) , mean(d) , atol = 0.1)
        m3 = Array{partype(d), 3}(undef, size(d)..., M)
        x = func[2](d, m3)
        @test x ≡ m3
        @test isapprox(mean(x) , mean(mean(d)) , atol = 0.1)
    end
    @testset "Testing matrix-variates with $key" for (key, func) in
        Dict("rand!(..., true)" => (dist, X) -> rand!(dist, X, true),
             "rand!(rng, true)" => (dist, X) -> rand!(rng, dist, X, true))
        m = Vector{Matrix{partype(d)}}(undef, M)
        x = func(d, m)
        @test x ≡ m
        @test isapprox(mean(x), mean(d) , atol = 0.1)
    end
    repeats = 10
    m = Vector{Matrix{partype(d)}}(undef, repeats)
    rand!(d, m)
    @test isassigned(m, 1)
    m1=m[1]
    rand!(d, m)
    @test m1 ≡ m[1]
    rand!(d, m, true)
    @test m1 ≢ m[1]
    m1 = m[1]
    rand!(d, m, false)
    @test m1 ≡ m[1]
    nothing
end

#  --------------------------------------------------
#  Check that the convert and partype methods work
#  --------------------------------------------------

function test_convert(d::MatrixDistribution)
    distname = getproperty(parentmodule(typeof(d)), nameof(typeof(d)))
    @test distname(params(d)...) == d
    @test d == deepcopy(d)
    for elty in (Float32, Float64, BigFloat)
        del1 = convert(distname{elty}, d)
        del2 = convert(distname{elty}, (Base.Fix1(getfield, d)).(fieldnames(typeof(d)))...)
        @test del1 isa distname{elty}
        @test del2 isa distname{elty}
        @test partype(del1) == elty
        @test partype(del2) == elty
        if elty === partype(d)
            @test del1 === d
        end
    end
    nothing
end

#  --------------------------------------------------
#  Check that cov and var agree
#  --------------------------------------------------

function test_cov(d::MatrixDistribution)
    @test vec(var(d)) ≈ diag(cov(d))
    @test reshape(cov(d), size(d)..., size(d)...) ≈ cov(d, Val(false))
end

test_cov(d::LKJ) = nothing

#  --------------------------------------------------
#  Check dim
#  --------------------------------------------------

function test_dim(d::MatrixDistribution)
    n = @test_deprecated(dim(d))
    @test n == size(d, 1)
    @test n == size(d, 2)
    @test n == size(mean(d), 1)
    @test n == size(mean(d), 2)
end

test_dim(d::Union{MatrixNormal, MatrixTDist}) = nothing

#  --------------------------------------------------
#  RUN EVERYTHING
#  --------------------------------------------------

function test_distr(d::MatrixDistribution, M::Integer)
    test_draw(d)
    test_draws(d, M)
    test_cov(d)
    test_convert(d)
    test_dim(d)
    nothing
end

function test_distr(dist::Type{<:MatrixDistribution},
                    n::Integer,
                    p::Integer,
                    M::Integer)
    d = dist(_rand_params(dist, Float64, n, p)...)
    test_distr(d, M)
    nothing
end

#  =============================================================================
#  2. test matrix-variate against the univariate it collapses to in the 1 x 1 case
#  =============================================================================

function test_against_univariate(D::MatrixDistribution, d::UnivariateDistribution)
    X = rand(D)
    x = X[1]
    @test logpdf(D, X) ≈ logpdf(d, x)
    @test pdf(D, X) ≈ pdf(d, x)
    @test mean(D)[1] ≈ mean(d)
    @test var(D)[1] ≈ var(d)
    nothing
end

function test_draws_against_univariate_cdf(D::MatrixDistribution, d::UnivariateDistribution)
    α = 0.025
    M = 100000
    matvardraws = [rand(D)[1] for m in 1:M]
    @test pvalue_kolmogorovsmirnoff(matvardraws, d) >= α
    nothing
end

test_draws_against_univariate_cdf(D::LKJ, d::UnivariateDistribution) = nothing

function test_against_univariate(dist::Type{<:MatrixDistribution})
    D = dist(_rand_params(dist, Float64, 1, 1)...)
    d = _univariate(D)
    test_against_univariate(D, d)
    test_draws_against_univariate_cdf(D, d)
    nothing
end

#  =============================================================================
#  3. test matrix-variate against the multivariate it collapses to in the row/column case
#  =============================================================================

function test_against_multivariate(D::MatrixDistribution, d::MultivariateDistribution)
    X = rand(D)
    x = vec(X)
    @test logpdf(D, X) ≈ logpdf(d, x)
    @test pdf(D, X) ≈ pdf(d, x)
    @test vec(mean(D)) ≈ mean(d)
    @test cov(D) ≈ cov(d)
    nothing
end

function test_against_multivariate(dist::Union{Type{MatrixNormal}, Type{MatrixTDist}}, n::Integer, p::Integer)
    D = dist(_rand_params(dist, Float64, n, 1)...)
    d = _multivariate(D)
    test_against_multivariate(D, d)
    D = dist(_rand_params(dist, Float64, 1, p)...)
    d = _multivariate(D)
    test_against_multivariate(D, d)
    nothing
end

test_against_multivariate(dist::Type{<:MatrixDistribution}, n::Integer, p::Integer) = nothing

#  =============================================================================
#  4. test density evaluation against archived output from Stan
#  =============================================================================

function jsonparams2dist(dist::Type{Wishart}, dict)
    ν = dict["params"][1][1]
    S = zeros(Float64, dict["dims"]...)
    S[:] = Vector{Float64}( dict["params"][2] )
    return Wishart(ν, S)
end

function jsonparams2dist(dist::Type{InverseWishart}, dict)
    ν = dict["params"][1][1]
    S = zeros(Float64, dict["dims"]...)
    S[:] = Vector{Float64}( dict["params"][2] )
    return InverseWishart(ν, S)
end

function jsonparams2dist(dist::Type{LKJ}, dict)
    d = dict["params"][1][1]
    η = dict["params"][2][1]
    return LKJ(d, η)
end

function unpack_matvar_json_dict(dist::Type{<:MatrixDistribution}, dict)
    d = jsonparams2dist(dist, dict)
    X = zeros(Float64, dict["dims"]...)
    X[:] = Vector{Float64}(dict["X"])
    lpdf = dict["lpdf"][1]
    return d, X, lpdf
end

function test_against_stan(dist::Type{<:MatrixDistribution})
    filename = joinpath(@__DIR__, "ref", "matrixvariates", "jsonfiles", "$(dist)_stan_output.json")
    stan_output = JSON.parsefile(filename)
    K = length(stan_output)
    for k in 1:K
        d, X, lpdf = unpack_matvar_json_dict(dist, stan_output[k])
        @test isapprox(logpdf(d, X), lpdf, atol = 1e-10)
        @test isapprox(logpdf(d, [X, X]), [lpdf, lpdf], atol=1e-8)
        @test isapprox(pdf(d, X), exp(lpdf), atol = 1e-6)
        @test isapprox(pdf(d, [X, X]), exp.([lpdf, lpdf]), atol=1e-6)
    end
    nothing
end

function test_against_stan(dist::Union{Type{MatrixNormal}, Type{MatrixTDist}, Type{MatrixBeta}, Type{MatrixFDist}})
    nothing
end

#  =============================================================================
#  5. special, distribution-specific tests
#  =============================================================================

test_special(dist::Type{<:MatrixDistribution}) = nothing

function test_special(dist::Type{MatrixNormal})
    D = MatrixNormal(_rand_params(MatrixNormal, Float64, 2, 2)...)
    @testset "X ~ MN(M, U, V) ⟺ vec(X) ~ N(vec(M), V ⊗ U)" begin
        d = vec(D)
        test_against_multivariate(D, d)
    end
    @testset "MatrixNormal mode" begin
        @test mode(D) == D.M
    end
    @testset "PDMat mixing and matching" begin
        n = 3
        p = 4
        M = randn(n, p)
        u = rand()
        U_scale = ScalMat(n, u)
        U_dense = Matrix(U_scale)
        U_pd    = PDMat(U_dense)
        U_pdiag = PDiagMat(u*ones(n))
        v = rand(p)
        V_pdiag = PDiagMat(v)
        V_dense = Matrix(V_pdiag)
        V_pd    = PDMat(V_dense)
        UV = kron(V_dense, U_dense)
        baseeval = logpdf(MatrixNormal(M, U_dense, V_dense), M)
        for U in [U_scale, U_dense, U_pd, U_pdiag]
            for V in [V_pdiag, V_dense, V_pd]
                d = MatrixNormal(M, U, V)
                @test cov(d) ≈ UV
                @test logpdf(d, M) ≈ baseeval
            end
        end
    end
    nothing
end

function test_special(dist::Type{Wishart})
    n = 3
    M = 5000
    α = 0.05
    ν, Σ = _rand_params(Wishart, Float64, n, n)
    d = Wishart(ν, Σ)
    H = rand(d, M)
    @testset "deprecations" begin
        for warn in (true, false)
            @test @test_deprecated(Wishart(n - 1, Σ, warn)) == Wishart(n - 1, Σ)
        end
    end
    @testset "meanlogdet" begin
        @test isapprox(Distributions.meanlogdet(d), mean(logdet.(H)), atol = 0.1)
    end
    @testset "H ~ W(ν, Σ), a ~ p(a) independent ⟹ a'Ha / a'Σa ~ χ²(ν)" begin
        q = MvTDist(10, randn(n), rand(d))
        ρ = Chisq(ν)
        A = rand(q, M)
        z = [A[:, m]'*H[m]*A[:, m] / (A[:, m]'*Σ*A[:, m]) for m in 1:M]
        @test pvalue_kolmogorovsmirnoff(z, ρ) >= α
    end
    @testset "H ~ W(ν, I) ⟹ H[i, i] ~ χ²(ν)" begin
        κ = n + 1
        ρ = Chisq(κ)
        g = Wishart(κ, ScalMat(n, 1))
        mymats = zeros(n, n, M)
        for m in 1:M
            mymats[:, :, m] = rand(g)
        end
        for i in 1:n
            @test pvalue_kolmogorovsmirnoff(mymats[i, i, :], ρ) >= α / n
        end
    end
    @testset "Check Singular Branch" begin
        # Check that no warnings are shown: #1410
        rank1 = @test_logs Wishart(n - 2, Σ)
        rank2 = @test_logs Wishart(n - 1, Σ)
        test_draw(rank1)
        test_draw(rank2)
        test_draws(rank1, rand(rank1, 10^6))
        test_draws(rank2, rand(rank2, 10^6))
        test_cov(rank1)
        test_cov(rank2)

        X = H[1]
        @test Distributions.singular_wishart_logkernel(d, X) ≈ Distributions.nonsingular_wishart_logkernel(d, X)
        @test Distributions.singular_wishart_logc0(n, ν, d.S, rank(d)) ≈ Distributions.nonsingular_wishart_logc0(n, ν, d.S)
        @test logpdf(d, X) ≈ Distributions.singular_wishart_logkernel(d, X) + Distributions.singular_wishart_logc0(n, ν, d.S, rank(d))
    end
    nothing
end

function test_special(dist::Type{InverseWishart})
    @testset "InverseWishart constructor" begin
        # Tests https://github.com/JuliaStats/Distributions.jl/issues/1948
        @test typeof(InverseWishart(5, ScalMat(5, 1))) == InverseWishart{Float64, ScalMat{Float64}}
        @test typeof(InverseWishart(5, PDiagMat(ones(Int, 5)))) == InverseWishart{Float64, PDiagMat{Float64, Vector{Float64}}}
    end
end

function test_special(dist::Type{MatrixTDist})
    @testset "MT(v, M, vΣ, Ω) → MN(M, Σ, Ω) as v → ∞" begin
        n, p = (6, 3)
        M, Σ, Ω = _rand_params(MatrixNormal, Float64, n, p)
        MT = MatrixTDist(1000, M, 1000Σ, Ω)
        MN = MatrixNormal(M, Σ, Ω)
        A = rand(MN)
        @test isapprox(logpdf(MT, A), logpdf(MN, A), atol = 0.1)
    end
    @testset "PDMat mixing and matching" begin
        n = 3
        p = 4
        ν = max(n, p) + 1
        M = randn(n, p)
        u = rand()
        U_scale = ScalMat(n, u)
        U_dense = Matrix(U_scale)
        U_pd    = PDMat(U_dense)
        U_pdiag = PDiagMat(u*ones(n))
        v = rand(p)
        V_pdiag = PDiagMat(v)
        V_dense = Matrix(V_pdiag)
        V_pd    = PDMat(V_dense)
        UV = kron(V_dense, U_dense) ./ (ν - 2)
        baseeval = logpdf(MatrixTDist(ν, M, U_dense, V_dense), M)
        for U in [U_scale, U_dense, U_pd, U_pdiag]
            for V in [V_pdiag, V_dense, V_pd]
                d = MatrixTDist(ν, M, U, V)
                @test cov(d) ≈ UV
                @test logpdf(d, M) ≈ baseeval
            end
        end
    end
    nothing
end

function test_special(dist::Type{LKJ})
    @testset "LKJ mode" begin
        @test mode(LKJ(5, 1.5)) == mean(LKJ(5, 1.5))
        @test_throws DomainError mode( LKJ(5, 0.5) )
    end
    @testset "LKJ marginals" begin
        d = 4
        η = abs(3randn())
        G = LKJ(d, η)
        M = 10000
        α = 0.05
        L = sum(1:(d - 1))
        ρ = Distributions._marginal(G)
        mymats = zeros(d, d, M)
        for m in 1:M
            mymats[:, :, m] = rand(G)
        end
        for i in 1:d
            for j in 1:i-1
                @test pvalue_kolmogorovsmirnoff(mymats[i, j, :], ρ) >= α / L
            end
        end
    end
    @testset "LKJ integrating constant" begin
        #  =============
        #  odd non-uniform
        #  =============
        d = 5
        η = 2.3
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
        #  =============
        #  odd uniform
        #  =============
        d = 5
        η = 1.0
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_odd(d, Float64)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_vine_loginvconst_uniform(d)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.corr_logvolume(d)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_odd(d, Float64)
        #  =============
        #  even non-uniform
        #  =============
        d = 6
        η = 2.3
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
        #  =============
        #  even uniform
        #  =============
        d = 6
        η = 1.0
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_even(d, Float64)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_vine_loginvconst_uniform(d)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_loginvconst_alt(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.corr_logvolume(d)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_even(d, Float64)
    end
    @testset "check integrating constant as a volume" begin
        #  d = 2: Lebesgue measure of the set of correlation matrices is 2.
        volume2D = 2
        @test volume2D ≈ exp( Distributions.lkj_onion_loginvconst(2, 1) )
        @test 1 / volume2D ≈ exp( LKJ(2, 1).logc0 )
        #  d = 3: Lebesgue measure of the set of correlation matrices is π²/2.
        #  See here: https://www.jstor.org/stable/2684832
        volume3D = 0.5π^2
        @test volume3D ≈ exp( Distributions.lkj_onion_loginvconst(3, 1) )
        @test 1 / volume3D ≈ exp( LKJ(3, 1).logc0 )
        #  d = 4: Lebesgue measure of the set of correlation matrices is (32/27)π².
        #  See here: https://doi.org/10.4169/amer.math.monthly.123.9.909
        volume4D = (32 / 27)*π^2
        @test volume4D ≈ exp( Distributions.lkj_onion_loginvconst(4, 1) )
        @test 1 / volume4D ≈ exp( LKJ(4, 1).logc0 )
    end
    nothing
end

#  =============================================================================
#  6. main method
#  =============================================================================

function test_matrixvariate(dist::Type{<:MatrixDistribution},
                            n::Integer,
                            p::Integer,
                            M::Integer)
    test_distr(dist, n, p, M)
    test_against_univariate(dist)
    test_against_multivariate(dist, n, p)
    test_against_stan(dist)
    test_special(dist)
    nothing
end

#  =============================================================================
#  7. run unit tests for matrix-variate distributions
#  =============================================================================

matrixvariates = [(MatrixNormal, 2, 4, 10^5),
                  (Wishart, 2, 2, 10^6),
                  (InverseWishart, 2, 2, 10^6),
                  (MatrixTDist, 2, 4, 10^5),
                  (MatrixBeta, 3, 3, 10^5),
                  (MatrixFDist, 3, 3, 10^5),
                  (LKJ, 3, 3, 10^5)]

for distribution in matrixvariates
    dist, n, p, M = distribution
    println("    testing $(dist)")
    @testset "$(dist)" begin
        test_matrixvariate(dist, n, p, M)
    end
end
end
