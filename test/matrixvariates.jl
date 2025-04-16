using Distributions
using Random
using LinearAlgebra
using PDMats
using Statistics
using Test
using StatsFuns
import JSON
import SpecialFunctions

# Test utility: Construct matrix distributions with random parameters

function _rand_dist(::Type{InverseWishart{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::InverseWishart{T} where {T<:Real}
    n == p || throw(ArgumentError("dims must be equal for InverseWishart"))
    rng = rng === nothing ? () : (rng,)
    ν = n + 3 + abs(10 * randn(rng..., T))
    X = rand(rng..., T, n, n)
    X .= 2 .* X .- 1
    Ψ = X * X'
    return InverseWishart(ν, Ψ)::InverseWishart{T}
end
function _rand_dist(::Type{LKJ{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::LKJ{T} where {T<:Real}
    n == p || throw(ArgumentError("dims must be equal for LKJ"))
    rng = rng === nothing ? () : (rng,)
    η = abs(3 * randn(rng..., T))
    return LKJ(n, η)
end
function _rand_dist(::Type{MatrixBeta{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::MatrixBeta{T} where {T<:Real}
    n == p || throw(ArgumentError("dims must be equal for MatrixBeta"))
    rng = rng === nothing ? () : (rng,)
    n1 = n + 1 + abs(10 * randn(rng..., T))
    n2 = n + 1 + abs(10 * randn(rng..., T))
    return MatrixBeta(n, n1, n2)
end
function _rand_dist(::Type{MatrixFDist{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::MatrixFDist{T} where {T<:Real}
    n == p || throw(ArgumentError("dims must be equal for MatrixFDist"))
    rng = rng === nothing ? () : (rng,)
    n1 = n + 1 + abs(10 * randn(rng..., T))
    n2 = n + 3 + abs(10 * randn(rng..., T))
    X = rand(rng..., T, n, n)
    X .= 2 .* X .- 1
    B = X * X'
    return MatrixFDist(n1, n2, B)
end
function _rand_dist(::Type{MatrixNormal{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::MatrixNormal{T} where {T<:Real}
    rng = rng === nothing ? () : (rng,)
    M = randn(rng..., T, n, p)
    X = rand(rng..., T, n, n)
    Y = rand(rng..., T, p, p)

    X .= 2 .* X .- 1
    U = X * X'

    Y .= 2 .* Y .- 1
    V = Y * Y'

    return MatrixNormal(M, U, V)
end
function _rand_dist(::Type{MatrixTDist{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::MatrixTDist{T} where {T<:Real}
    rng = rng === nothing ? () : (rng,)
    ν = n + p + 1 + abs(10 * randn(rng..., T))
    M = randn(rng..., T, n, p)
    X = rand(rng..., T, n, n)
    Y = rand(rng..., T, p, p)

    X .= 2 .* X .- 1
    Σ = X * X'

    Y .= 2 .* Y .- 1
    Ω = Y * Y'

    return MatrixTDist(ν, M, Σ, Ω)
end
function _rand_dist(::Type{Wishart{T}}, n::Int, p::Int; rng::Union{AbstractRNG,Nothing} = nothing)::Wishart{T} where {T<:Real}
    n == p || throw(ArgumentError("dims must be equal for Wishart"))
    rng = rng === nothing ? () : (rng,)
    ν = n - 1 + abs(10 * randn(rng..., T))
    X = rand(rng..., T, n, n)
    X .= 2 .* X .- 1
    S = X * X'
    return Wishart(ν, S)
end

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

function test_draw(d::MatrixDistribution; rng::Union{AbstractRNG,Nothing}=nothing)
    X = rng === nothing ? rand(d) : rand(rng, d)
    @test X isa Matrix{float(partype(d))}
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

#  --------------------------------------------------
#  Check that sample quantities are close to population quantities
#  --------------------------------------------------

function test_draws(d::MatrixDistribution, draws::AbstractArray{<:AbstractMatrix})
    @test mean(draws) ≈ mean(d) rtol = 0.01
    draws_matrix = mapreduce(vec, hcat, draws)
    @test cov(draws_matrix; dims=2) ≈ cov(d) rtol = 0.5
    nothing
end

function test_draws(d::LKJ, draws::AbstractArray{<:AbstractMatrix})
    @test isapprox(mean(draws), mean(d), atol = 0.1)
    @test isapprox(var(draws), var(d) , atol = 0.1)
    nothing
end

function test_draws(d::MatrixDistribution, M::Integer; rng::Union{AbstractRNG,Nothing} = nothing)
    rng = rng === nothing ? () : (rng,)
    @testset "Testing matrix-variates with rand(...)" begin
        test_draws(d, rand(rng..., d, M))
    end
    @testset "Testing matrix-variates with rand!(..., false)" begin
        m = [Matrix{float(partype(d))}(undef, size(d)) for _ in 1:M]
        x = rand!(rng..., d, m, false)
        @test x ≡ m
        @test mean(x) ≈ mean(d) rtol = 0.1
        m3 = Array{float(partype(d))}(undef, size(d)..., M)
        x = rand!(rng..., d, m3)
        @test x ≡ m3
        @test dropdims(mean(x; dims=3); dims=3) ≈ mean(d) rtol = 0.1
    end
    @testset "Testing matrix-variates with rand!(..., true)" begin
        m = Vector{Matrix{float(partype(d))}}(undef, M)
        x = rand!(rng..., d, m, true)
        @test x ≡ m
        @test mean(x) ≈ mean(d) rtol = 0.1
    end
    repeats = 10
    m = Vector{Matrix{float(partype(d))}}(undef, repeats)
    rand!(rng..., d, m)
    @test isassigned(m, 1)
    m1=m[1]
    rand!(rng..., d, m)
    @test m1 ≡ m[1]
    rand!(rng..., d, m, true)
    @test m1 ≢ m[1]
    m1 = m[1]
    rand!(rng..., d, m, false)
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

test_cov(::LKJ) = nothing

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

test_dim(::Union{MatrixNormal, MatrixTDist}) = nothing

#  --------------------------------------------------
#  RUN EVERYTHING
#  --------------------------------------------------

function test_distr(d::MatrixDistribution, M::Integer; rng::Union{AbstractRNG,Nothing} = nothing)
    test_draw(d; rng=rng)
    test_draws(d, M; rng=rng)
    test_cov(d)
    test_convert(d)
    test_dim(d)
    nothing
end

function test_distr(::Type{D},
                    n::Integer,
                    p::Integer,
                    M::Integer;
                    rng::Union{AbstractRNG,Nothing} = nothing) where {D<:MatrixDistribution}
    d = _rand_dist(D, n, p; rng=rng)
    test_distr(d, M; rng=rng)
    nothing
end

#  =============================================================================
#  2. test matrix-variate against the univariate it collapses to in the 1 x 1 case
#  =============================================================================

function _univariate(d::InverseWishart)
    ν, Ψ = params(d)
    α = ν / 2
    β = Matrix(Ψ)[1] / 2
    return InverseGamma(α, β)
end
function _univariate(d::MatrixBeta)
    p, n1, n2 = params(d)
    return Beta(n1 / 2, n2 / 2)
end
function _univariate(d::MatrixFDist)
    n1, n2, B = params(d)
    μ = zero(partype(d))
    σ = (n1 / n2) * Matrix(B)[1]
    return μ + σ * FDist(n1, n2)
end
function _univariate(d::MatrixNormal)
    M, U, V = params(d)
    μ = M[1]
    σ = sqrt( Matrix(U)[1] * Matrix(V)[1] )
    return Normal(μ, σ)
end
function _univariate(d::MatrixTDist)
    ν, M, Σ, Ω = params(d)
    μ = M[1]
    σ = sqrt( Matrix(Σ)[1] * Matrix(Ω)[1] / ν )
    return μ + σ * TDist(ν)
end
function _univariate(d::Wishart)
    df, S = params(d)
    α = df / 2
    β = 2 * first(S)
    return Gamma(α, β)
end

function test_against_univariate(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {D<:MatrixDistribution}
    dist = _rand_dist(D, 1, 1; rng=rng)
    univariate_dist = _univariate(dist)
    rng = rng === nothing ? () : (rng,)
    @testset "Univariate Case: (log)pdf, mean, and var" begin
        X = rand(rng..., dist)
        x = X[1]
        @test logpdf(dist, X) ≈ logpdf(univariate_dist, x)
        @test pdf(dist, X) ≈ pdf(univariate_dist, x)
        @test mean(dist)[1] ≈ mean(univariate_dist)
        @test var(dist)[1] ≈ var(univariate_dist)
    end
    @testset "Univariate Case: Hypothesis test" begin
        α = 0.025
        M = 10_000
        matvardraws = [first(rand(rng..., dist)) for _ in 1:M]
        @test pvalue_kolmogorovsmirnoff(matvardraws, univariate_dist) > α
    end
    nothing
end

function test_against_univariate(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:LKJ{T}}
    dist = _rand_dist(D, 1, 1; rng=rng)
    X = rng === nothing ? rand(dist) : rand(rng, dist)
    @test isone(first(X))
    nothing
end

#  =============================================================================
#  3. test matrix-variate against the multivariate it collapses to in the row/column case
#  =============================================================================

function _multivariate(d::MatrixNormal)
    n, p = size(d)
    if n > 1 && p > 1
        throw(ArgumentError("Row or col dim of `MatrixNormal` must be 1 to coerce to `MvNormal`"))
    end
    return vec(d)
end
_multivariate(d::MatrixTDist) = MvTDist(d)

function test_against_multivariate(::Type{D}, n::Integer, p::Integer; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:Union{MatrixNormal{T}, MatrixTDist{T}}}
    dist = _rand_dist(D, n, 1; rng=rng)
    multivariate_dist = _multivariate(dist)
    X = rng === nothing ? rand(dist) : rand(rng, dist)
    x = vec(X)
    @test logpdf(dist, X) ≈ logpdf(multivariate_dist, x)
    @test pdf(dist, X) ≈ pdf(multivariate_dist, x)
    @test vec(mean(dist)) ≈ mean(multivariate_dist)
    @test cov(dist) ≈ cov(multivariate_dist)
    nothing
end

test_against_multivariate(::Type{<:MatrixDistribution}, n::Integer, p::Integer; rng::Union{AbstractRNG,Nothing} = nothing) = nothing

#  =============================================================================
#  4. test density evaluation against archived output from Stan
#  =============================================================================

function jsonparams2dist(::Type{<:Wishart}, dict)
    ν = dict["params"][1][1]
    S = zeros(Float64, dict["dims"]...)
    S[:] = Vector{Float64}( dict["params"][2] )
    return Wishart(ν, S)
end

function jsonparams2dist(::Type{<:InverseWishart}, dict)
    ν = dict["params"][1][1]
    S = zeros(Float64, dict["dims"]...)
    S[:] = Vector{Float64}( dict["params"][2] )
    return InverseWishart(ν, S)
end

function jsonparams2dist(::Type{<:LKJ}, dict)
    d = dict["params"][1][1]
    η = dict["params"][2][1]
    return LKJ(d, η)
end

function unpack_matvar_json_dict(::Type{D}, dict) where {D<:Union{Wishart,InverseWishart,LKJ}}
    d = jsonparams2dist(D, dict)
    X = zeros(Float64, dict["dims"]...)
    X[:] = Vector{Float64}(dict["X"])
    lpdf = dict["lpdf"][1]
    return d, X, lpdf
end

function test_against_stan(::Type{D}) where {D<:Union{Wishart,InverseWishart,LKJ}}
    filename = joinpath(@__DIR__, "ref", "matrixvariates", "jsonfiles", "$(nameof(D))_stan_output.json")
    stan_output = JSON.parsefile(filename)
    for stan_output_i in stan_output
        d, X, lpdf = unpack_matvar_json_dict(D, stan_output_i)
        @test logpdf(d, X) ≈ lpdf atol = 1e-10
        @test logpdf(d, [X, X]) ≈ [lpdf, lpdf] atol=1e-8
        @test pdf(d, X) ≈ exp(lpdf) atol = 1e-6
        @test pdf(d, [X, X]) ≈ exp.([lpdf, lpdf]) atol=1e-6
    end
    nothing
end

test_against_stan(::Type{<:MatrixDistribution}) = nothing

#  =============================================================================
#  5. special, distribution-specific tests
#  =============================================================================

test_special(::Type{<:MatrixDistribution}; rng::Union{AbstractRNG,Nothing} = nothing) = nothing

function test_special(::Type{DT}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,DT<:MatrixNormal{T}}
    D = _rand_dist(DT, 2, 2; rng=rng)
    @testset "X ~ MN(M, U, V) ⟺ vec(X) ~ N(vec(M), V ⊗ U)" begin
        test_against_multivariate(DT, 2, 2; rng=rng)
    end
    @testset "MatrixNormal mode" begin
        @test mode(D) == D.M
    end
    @testset "PDMat mixing and matching" begin
        n = 3
        p = 4
        rng = rng === nothing ? () : (rng,)
        M = randn(rng..., n, p)
        u = rand(rng...)
        U_scale = ScalMat(n, u)
        U_dense = Matrix(U_scale)
        U_pd    = PDMat(U_dense)
        U_pdiag = PDiagMat(u*ones(n))
        v = rand(rng..., p)
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

function test_special(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:Wishart{T}}
    n = 3
    M = 5000
    α = 0.05
    d = _rand_dist(D, n, n; rng=rng)
    ν, Σ = params(d)
    H = rng === nothing ? rand(d, M) : rand(rng, d, M)
    @testset "deprecations" begin
        for warn in (true, false)
            @test @test_deprecated(Wishart(n - 1, Σ, warn)) == Wishart(n - 1, Σ)
        end
    end
    @testset "meanlogdet" begin
        # Errors for Float32?!
        if T === Float64
            @test Distributions.meanlogdet(d) ≈ mean(logdet, H) rtol = 0.1
        end
    end
    @testset "H ~ W(ν, Σ), a ~ p(a) independent ⟹ a'Ha / a'Σa ~ χ²(ν)" begin
        q = rng === nothing ? MvTDist(10, randn(T, n), rand(d)) : MvTDist(10, randn(rng, T, n), rand(rng, d))
        ρ = Chisq(ν)
        A = rng === nothing ? rand(q, M) : rand(rng, q, M)
        z = [A[:, m]'*H[m]*A[:, m] / (A[:, m]'*Σ*A[:, m]) for m in 1:M]
        @test pvalue_kolmogorovsmirnoff(z, ρ) >= α
    end
    @testset "H ~ W(ν, I) ⟹ H[i, i] ~ χ²(ν)" begin
        κ = n + 1
        ρ = Chisq(κ)
        g = Wishart(T(κ), ScalMat(n, T(1)))
        mymats = Array{T}(undef, n, n, M)
        for m in 1:M
            mymats[:, :, m] = rng === nothing ? rand(g) : rand(rng, g)
        end
        # Compute p-values of the KS tests for each diagonal entry
        pvalues = map(1:n) do i
            pvalue_kolmogorovsmirnoff(@view(mymats[i, i, :]), ρ)
        end
        # Perform test with Bonferroni-Holm correction
        sort!(pvalues)
        for (i, p) in enumerate(pvalues)
            @test p > α / (n + 1 - i)
        end
    end
    @testset "Check Singular Branch" begin
        # Check that no warnings are shown: #1410
        for ν in (n - 2, n - 1)
            dist = @test_logs Wishart(ν, Σ)
            test_draw(dist; rng=rng)
            test_draws(dist, rng === nothing ? rand(dist, 10^6) : rand(rng, dist, 10^6))
            test_cov(dist)
        end

        X = H[1]
        @test Distributions.singular_wishart_logkernel(d, X) ≈ Distributions.nonsingular_wishart_logkernel(d, X)
        @test Distributions.singular_wishart_logc0(n, ν, d.S, rank(d)) ≈ Distributions.nonsingular_wishart_logc0(n, ν, d.S)
        if T === Float64
            @test logpdf(d, X) ≈ Distributions.singular_wishart_logkernel(d, X) + Distributions.singular_wishart_logc0(n, ν, d.S, rank(d))
        end
    end
    nothing
end

function test_special(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:InverseWishart{T}}
    @testset "InverseWishart constructor" begin
        # Tests https://github.com/JuliaStats/Distributions.jl/issues/1948
        @test InverseWishart(5, ScalMat(5, T(1))) isa InverseWishart{T, ScalMat{T}}
        @test InverseWishart(5, PDiagMat(ones(T, 5))) isa InverseWishart{T, PDiagMat{T, Vector{T}}}
    end
end
                                                                                                            
function test_special(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:MatrixTDist{T}}
    @testset "MT(v, M, vΣ, Ω) → MN(M, Σ, Ω) as v → ∞" begin
        n, p = (6, 3)
        MN = _rand_dist(MatrixNormal{T}, n, p; rng=rng)
        A = rng === nothing ? rand(MN) : rand(rng, MN)
        M, Σ, Ω = params(MN)
        MT = MatrixTDist(1000, M, 1000Σ, Ω)
        @test logpdf(MT, A) ≈ logpdf(MN, A) atol = 0.1
    end
    @testset "PDMat mixing and matching" begin
        n = 3
        p = 4
        ν = max(n, p) + 1
        M = rng === nothing ? randn(T, n, p) : randn(rng, T, n, p)
        u = rng === nothing ? rand(T) : rand(rng, T)
        U_scale = ScalMat(n, u)
        U_dense = Matrix(U_scale)
        U_pd    = PDMat(U_dense)
        U_pdiag = PDiagMat(u*ones(n))
        v = rng === nothing ? rand(T, p) : rand(rng, T, p)
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

# Equation after (16) in LKJ (2009 JMA)
function lkj_vine_loginvconst_uniform(d::Integer, η::Real)
    @assert isone(η)
    expsum = betasum = zero(float(first(promote(d, η))))
    for k in 1:(d - 1)
        α = oftype(betasum, k + 1) / 2
        expsum += k^2
        betasum += k * SpecialFunctions.logbeta(α, α)
    end
    loginvconst = expsum * logtwo + betasum
    return loginvconst
end
# Third line in first proof of Section 3.3 in LKJ (2009 JMA)
function lkj_loginvconst_alt(d::Integer, η::Real)
    loginvconst = float(zero(η))
    halflogπ = oftype(loginvconst, logπ) / 2
    z = SpecialFunctions.loggamma(η + oftype(loginvconst, d - 1) / 2)
    for k in 1:(d - 1)
        loginvconst += k * halflogπ + SpecialFunctions.loggamma(η + oftype(loginvconst, d - 1 - k) / 2) - z
    end
    return loginvconst
end
# https://doi.org/10.4169/amer.math.monthly.123.9.909
function corr_logvolume(n::Integer, η::Real)
    logvol = zero(float(first(promote(n, η))))
    halflogπ = oftype(logvol, logπ) / 2
    for k in 1:(n - 1)
        logvol += k * (halflogπ + SpecialFunctions.loggamma(oftype(logvol, k+1)/2) - SpecialFunctions.loggamma(oftype(logvol, k+2)/2))
    end
    return logvol
end

function test_special(::Type{D}; rng::Union{AbstractRNG,Nothing} = nothing) where {T<:Real,D<:LKJ{T}}
    @testset "LKJ mode" begin
        @test mode(LKJ(5, T(3//2))) == mean(LKJ(5, T(3//2)))
        @test_throws DomainError mode(LKJ(5, T(1//2)))
    end
    @testset "LKJ marginals" begin
        d = 4
        G = _rand_dist(D, d, d; rng=rng)
        M = 10000
        α = 0.025
        L = (d * (d - 1)) >> 1
        ρ = Distributions._marginal(G)
        mymats = Array{T}(undef, d, d, M)
        for m in 1:M
            mymats[:, :, m] = rng === nothing ? rand(G) : rand(rng, G)
        end
        # Compute p-values of the KS tests for each lower-triangular entry
        pvalues = Vector{Float64}(undef, L)
        k = 0
        for j in 1:d
            for i in (j+1):d
                pvalues[k += 1] = pvalue_kolmogorovsmirnoff(@view(mymats[i, j, :]), ρ)
            end
        end
        # Perform test with Bonferroni-Holm correction
        sort!(pvalues)
        for (i, p) in enumerate(pvalues)
            @test p > α / (L + 1 - i)
        end
    end
    @testset "LKJ integrating constant" begin
        #  =============
        #  odd non-uniform
        #  =============
        d = 5
        η = T(23//10)
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ lkj_loginvconst_alt(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
        #  =============
        #  odd uniform
        #  =============
        d = 5
        η = T(1)
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_odd(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ lkj_vine_loginvconst_uniform(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ lkj_loginvconst_alt(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ corr_logvolume(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_odd(d, η)
        #  =============
        #  even non-uniform
        #  =============
        d = 6
        η = T(23//10)
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ lkj_loginvconst_alt(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst(d, η)
        #  =============
        #  even uniform
        #  =============
        d = 6
        η = T(1)
        lkj = LKJ(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ Distributions.lkj_onion_loginvconst_uniform_even(d, η)
        @test Distributions.lkj_vine_loginvconst(d, η) ≈ lkj_vine_loginvconst_uniform(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ lkj_loginvconst_alt(d, η)
        @test Distributions.lkj_onion_loginvconst(d, η) ≈ corr_logvolume(d, η)
        @test lkj.logc0 == -Distributions.lkj_onion_loginvconst_uniform_even(d, η)
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

function test_matrixvariate(::Type{D},
                            n::Integer,
                            p::Integer,
                            M::Integer;
                            rng::Union{AbstractRNG,Nothing} = nothing) where {D<:MatrixDistribution}
    test_distr(D, n, p, M; rng=rng)
    test_against_univariate(D; rng=rng)
    test_against_multivariate(D, n, p; rng=rng)
    test_against_stan(D)
    test_special(D; rng=rng)
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
for (D, n, p, M) in matrixvariates
    println("    testing $(D)")
    @testset "$(D) ($(rng === nothing ? "with" : "without") RNG)" for rng in (nothing, Random.default_rng())
        @testset for T in (Float32, Float64)
            test_matrixvariate(D{T}, n, p, M; rng=rng)
        end
    end
end
end
