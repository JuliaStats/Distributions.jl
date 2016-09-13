# Tests on Multivariate LogNormal distributions

using Distributions, Compat
using Base.Test


####### Core testing procedure

function test_mvlognormal(g::MvLogNormal, n_tsamples::Int=10^6)
    d = length(g)
    mn = mean(g)
    md = median(g)
    mo = mode(g)
    S = cov(g)
    s = var(g)
    e = entropy(g)
    @test partype(g) == Float64
    @test isa(mn, Vector{Float64})
    @test isa(md, Vector{Float64})
    @test isa(mo, Vector{Float64})
    @test isa(s, Vector{Float64})
    @test isa(S, Matrix{Float64})
    @test length(mn) == d
    @test length(md) == d
    @test length(mo) == d
    @test length(s) == d
    @test size(S) == (d, d)
    @test_approx_eq s diag(S)
    @test_approx_eq md @compat(exp.(mean(g.normal)))
    @test_approx_eq mn @compat(exp.(mean(g.normal) + var(g.normal)/2))
    @test_approx_eq mo @compat(exp.(mean(g.normal) - var(g.normal)))
    @test_approx_eq entropy(g) d*(1 + Distributions.log2π)/2 + logdetcov(g.normal)/2 + sum(mean(g.normal))
    gg = typeof(g)(MvNormal(params(g)...))
    @test full(g.normal.μ) == full(gg.normal.μ)
    @test full(g.normal.Σ) == full(gg.normal.Σ)
    @test insupport(g,ones(d))
    @test !insupport(g,zeros(d))
    @test !insupport(g,-ones(d))

    # sampling
    X = rand(g, n_tsamples)
    emp_mn = vec(mean(X, 2))
    emp_md = vec(median(X, 2))
    Z = X .- emp_mn
    emp_cov = A_mul_Bt(Z, Z) * (1.0 / n_tsamples)
    for i = 1:d
        @test_approx_eq_eps emp_mn[i] mn[i] (sqrt(s[i] / n_tsamples) * 8.0)
    end
    for i = 1:d
        @test_approx_eq_eps emp_md[i] md[i] (sqrt(s[i] / n_tsamples) * 8.0)
    end
    for i = 1:d, j = 1:d
        @test_approx_eq_eps emp_cov[i,j] S[i,j] (sqrt(s[i] * s[j]) * 20.0) / sqrt(n_tsamples)
    end

    # evaluation of logpdf and pdf
    for i = 1:min(100, n_tsamples)
        @test_approx_eq logpdf(g, X[:,i]) log(pdf(g, X[:,i]))
    end
    @test_approx_eq logpdf(g, X) @compat(log.(pdf(g, X)))
    @test isequal(logpdf(g, zeros(d)),-Inf)
    @test isequal(logpdf(g, -mn),-Inf)
    @test isequal(pdf(g, zeros(d)),0.0)
    @test isequal(pdf(g, -mn),0.0)

    # test the location and scale functions
    @test_approx_eq_eps location(g) location(MvLogNormal,:meancov,mean(g),cov(g)) 1e-8
    @test_approx_eq_eps location(g) location(MvLogNormal,:mean,mean(g),scale(g)) 1e-8
    @test_approx_eq_eps location(g) location(MvLogNormal,:median,median(g),scale(g)) 1e-8
    @test_approx_eq_eps location(g) location(MvLogNormal,:mode,mode(g),scale(g)) 1e-8
    @test_approx_eq_eps scale(g) scale(MvLogNormal,:meancov,mean(g),cov(g)) 1e-8

    @test_approx_eq_eps location(g) location!(MvLogNormal,:meancov,mean(g),cov(g),zeros(mn)) 1e-8
    @test_approx_eq_eps location(g) location!(MvLogNormal,:mean,mean(g),scale(g),zeros(mn)) 1e-8
    @test_approx_eq_eps location(g) location!(MvLogNormal,:median,median(g),scale(g),zeros(mn)) 1e-8
    @test_approx_eq_eps location(g) location!(MvLogNormal,:mode,mode(g),scale(g),zeros(mn)) 1e-8
    @test_approx_eq_eps scale(g) scale!(MvLogNormal,:meancov,mean(g),cov(g),zeros(S)) 1e-8

    lc1,sc1 = params(MvLogNormal,mean(g),cov(g))
    lc2,sc2 = params!(MvLogNormal,mean(g),cov(g),similar(mn),similar(S))
    @test_approx_eq_eps location(g) lc1  1e-8
    @test_approx_eq_eps location(g) lc2  1e-8
    @test_approx_eq_eps scale(g) sc1  1e-8
    @test_approx_eq_eps scale(g) sc2  1e-8
end

####### Validate results for a single-dimension MvLogNormal by comparing with univariate LogNormal
println("    comparing results from MvLogNormal with univariate LogNormal")
l1 = LogNormal(0.1,0.4)
l2 = MvLogNormal(0.1*ones(1),0.4)
@test_approx_eq [mean(l1)] mean(l2)
@test_approx_eq [median(l1)] median(l2)
@test_approx_eq [mode(l1)] mode(l2)
@test_approx_eq [var(l1)] var(l2)
@test_approx_eq [entropy(l1)] entropy(l2)
@test_approx_eq logpdf(l1,5.0) logpdf(l2,[5.0])
@test_approx_eq pdf(l1,5.0) pdf(l2,[5.0])
@test (srand(78393) ; [rand(l1)]) == (srand(78393) ; rand(l2))

###### General Testing

mu = [0.1, 0.2, 0.3]
va = [0.16, 0.25, 0.36]
C = [0.4 -0.2 -0.1; -0.2 0.5 -0.1; -0.1 -0.1 0.6]

for (g, μ, Σ) in [
    (MvLogNormal(mu,PDMats.PDMat(C)), mu, C),
    (MvLogNormal(PDMats.PDiagMat(Vector{Float64}(@compat(sqrt.(va))))), zeros(3), diagm(va)), # Julia 0.4 loses type information so Vector{Float64} can be dropped when we don't support 0.4
    (MvLogNormal(mu, sqrt(0.2)), mu, 0.2 * eye(3)),
    (MvLogNormal(3, sqrt(0.2)), zeros(3), 0.2 * eye(3)),
    (MvLogNormal(mu, Vector{Float64}(@compat(sqrt.(va)))), mu, diagm(va)), # Julia 0.4 loses type information so Vector{Float64} can be dropped when we don't support 0.4
    (MvLogNormal(Vector{Float64}(@compat(sqrt.(va)))), zeros(3), diagm(va)), # Julia 0.4 loses type information so Vector{Float64} can be dropped when we don't support 0.4
    (MvLogNormal(mu, C), mu, C),
    (MvLogNormal(C), zeros(3), C) ]

    println("    testing $(typeof(g)) with normal distribution $(Distributions.distrname(g.normal))")

    m,s = params(g)
    @test_approx_eq full(m) μ
    test_mvlognormal(g, 10^4)
end

##### Constructors and conversions
d = MvLogNormal(Array{Float32}(mu), PDMats.PDMat(Array{Float32}(C)))
@test typeof(convert(MvLogNormal{Float64}, d)) == typeof(MvLogNormal(mu, PDMats.PDMat(C)))
@test typeof(convert(MvLogNormal{Float64}, d.normal.μ, d.normal.Σ)) == typeof(MvLogNormal(mu, PDMats.PDMat(C)))
