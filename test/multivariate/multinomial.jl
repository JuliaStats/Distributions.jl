# Tests for Multinomial

using Distributions, Random, StaticArrays
using Test


@testset "testing Multinomial with type $T" for T in (Float16, Float32, Float64)
    p = T[0.2, 0.5, 0.3]
    nt = 10
    rng = MersenneTwister(123)
    d = Multinomial(nt, p)

    @testset "Testing Multinomial with $key" for (key, func) in
        Dict("rand(...)" => [rand, rand],
             "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])
    
    # Basics
    
    @test length(d) == 3
    @test d.n == nt
    @test mean(d) ≈ T[2., 5., 3.]
    @test var(d)  ≈ T[1.6, 2.5, 2.1]
    @test cov(d)  ≈ T[1.6 -1.0 -0.6; -1.0 2.5 -1.5; -0.6 -1.5 2.1]
    
    @test insupport(d, [1, 6, 3])
    @test !insupport(d, [2, 6, 3])
    @test partype(d) === T
    
    # Conversion
    @test typeof(d) === Multinomial{T, Vector{T}}
    for S in (Float16, Float32, Float64)
        S === T && continue
        @test typeof(convert(Multinomial{S}, d)) === Multinomial{S, Vector{S}}
        @test typeof(convert(Multinomial{S, Vector{S}}, d)) === Multinomial{S, Vector{S}}
        @test typeof(convert(Multinomial{S}, params(d)...)) === Multinomial{S, Vector{S}}
        @test typeof(convert(Multinomial{S, Vector{S}}, params(d)...)) == Multinomial{S, Vector{S}}
    end
    @test convert(Multinomial{T}, d) === d
    @test convert(Multinomial{T, Vector{T}}, d) === d
    
    # random sampling
    
    x = @inferred(func[1](d))
    @test isa(x, Vector{Int})
    @test sum(x) == nt
    @test insupport(d, x)
    @test size(x) == size(d)
    @test length(x) == length(d)
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)
    
    x = @inferred(func[2](d, 100))
    @test isa(x, Matrix{Int})
    @test all(sum(x, dims=1) .== nt)
    @test all(insupport(d, x))
    
    x = @inferred(func[1](sampler(d)))
    @test isa(x, Vector{Int})
    @test sum(x) == nt
    @test insupport(d, x)
    
    # logpdf
    
    x1 = [1, 6, 3]
    
    @test @inferred(pdf(d, x1)) ≈ T(0.070875)
    @test @inferred(logpdf(d, x1)) ≈ log(pdf(d, x1))
    
    x = func[2](d, 100)
    pv = pdf(d, x)
    lp = logpdf(d, x)
    for i in 1 : size(x, 2)
        @test pv[i] ≈ pdf(d, x[:,i])
        @test lp[i] ≈ logpdf(d, x[:,i])
    end
    
    # test result type of logpdf
    @test typeof(logpdf(d, x1)) === T
    
    # test degenerate cases of logpdf
    d1 = Multinomial(1, T[0.5, 0.5, 0.0])
    d2 = Multinomial(0, T[0.5, 0.5, 0.0])
    x2 = [1, 0, 0]
    x3 = [0, 0, 1]
    x4 = [1, 0, 1]
    
    @test logpdf(d1, x2) ≈ T(log(0.5))
    @test logpdf(d2, x2) == -Inf
    @test logpdf(d1, x3) == -Inf
    @test logpdf(d2, x3) == -Inf
    
    # suffstats
    
    d0 = d
    n0 = 100
    x = func[2](d0, n0)
    w = func[1](n0)
    
    ss = suffstats(Multinomial, x)
    @test isa(ss, Distributions.MultinomialStats)
    @test ss.n == nt
    @test ss.scnts == vec(sum(T[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)], dims=2))
    @test ss.tw == n0
    
    ss = suffstats(Multinomial, x, w)
    @test isa(ss, Distributions.MultinomialStats)
    @test ss.n == nt
    @test ss.scnts ≈ T[x[i,j] for i = 1:size(x,1), j = 1:size(x,2)] * w
    @test ss.tw    ≈ sum(w)
    
    # fit
    
    x = func[2](d0, 10^5)
    @test size(x) == (length(d0), 10^5)
    @test all(sum(x, dims=1) .== nt)
    
    r = fit(Multinomial, x)
    @test r.n == nt
    @test length(r) == length(p)
    @test probs(r) ≈ p atol=0.02
    r = fit(Multinomial{Float64}, x)
    @test r.n == nt
    @test length(r) == length(p)
    @test probs(r) ≈ p atol=0.02
    
    r = fit_mle(Multinomial, x, fill(2.0, size(x,2)))
    @test r.n == nt
    @test length(r) == length(p)
    @test probs(r) ≈ p atol=0.02
    
    # behavior for n = 0
    d0 = Multinomial(0, p)
    @test func[1](d0) == [0, 0, 0]
    @test pdf(d0, [0, 0, 0]) == 1
    @test pdf(d0, [0, 1, 0]) == 0
    @test mean(d0) == [0, 0, 0]
    @test var(d0) == [0, 0, 0]
    @test cov(d0) == zeros(3, 3)
    @test entropy(d0) == 0
    @test insupport(d0, [0, 0, 0]) == true
    @test insupport(d0, [0, 0, 4]) == false
    @test length(d0) == 3
    @test size(d0) == (3,)
    
    # Abstract vector p
    
    @test typeof(Multinomial(nt, SVector{length(p), T}(p))) == Multinomial{T, SVector{3, T}}
    
    end
    
    @testset "Testing Multinomial with $key" for (key, func) in
        Dict("rand!(...)" => (dist, X) -> rand!(dist, X),
             "rand!(rng, ...)" => (dist, X) -> rand!(rng, dist, X))
        # random sampling
        X = Matrix{Int}(undef, length(p), 100)
        x = @inferred(func(d, X))
        @test x ≡ X
        @test isa(x, Matrix{Int})
        @test all(sum(x, dims=1) .== nt)
        @test all(insupport(d, x))
        pv = pdf(d, x)
        lp = logpdf(d, x)
        for i in 1 : size(x, 2)
            @test pv[i] ≈ pdf(d, x[:,i])
            @test lp[i] ≈ logpdf(d, x[:,i])
        end
    end
    
    @testset "Testing Multinomial with $key" for (key, func) in
        Dict("rand!(..., true)" => (dist, X) -> rand!(dist, X, true),
             "rand!(rng, ..., true)" => (dist, X) -> rand!(rng, dist, X, true))
        # random sampling
        X = Vector{Vector{Int}}(undef, 100)
        x = @inferred(func(d, X))
        @test x ≡ X
        @test all(sum.(x) .== nt)
        @test all(insupport(d, a) for a in x)
    end
    
    @testset "Testing Multinomial with $key" for (key, func) in
        Dict("rand!(..., false)" => (dist, X) -> rand!(dist, X, false),
             "rand!(rng, ..., false)" => (dist, X) -> rand!(rng, dist, X, false))
        # random sampling
        X = [Vector{Int}(undef, length(p)) for _ in Base.OneTo(100)]
        x1 = X[1]
        x = @inferred(func(d, X))
        @test x1 ≡ X[1]
        @test all(sum.(x) .== nt)
        @test all(insupport(d, a) for a in x)
    end
    
    repeats = 10
    m = Vector{Vector{T}}(undef, repeats)
    @inferred(rand!(d, m))
    @test isassigned(m, 1)
    m1=m[1]
    @inferred(rand!(d, m))
    @test m1 ≡ m[1]
    @inferred(rand!(d, m, true))
    @test m1 ≢ m[1]
    m1 = m[1]
    @inferred(rand!(d, m, false))
    @test m1 ≡ m[1]
    
    p = T[0.2, 0.4, 0.3, 0.1]
    nt = 10
    d = Multinomial(nt, p)
    @test_throws DimensionMismatch rand!(d, m, false)
    @test_nowarn rand!(d, m)
    
    p_v = T[0.1, 0.4, 0.3, 0.8]
    @test_throws DomainError Multinomial(10, p_v)
    @test_throws DomainError Multinomial(10, p_v; check_args=true)
    Multinomial(10, p_v; check_args=false) # should not warn
end
