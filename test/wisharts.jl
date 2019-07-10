using Distributions, Random
using Test, LinearAlgebra


v = 7.0
S = Matrix(1.0I, 2, 2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(v,S)
IW = InverseWishart(v,S)

rng = MersenneTwister(123)

@testset "Testing matrix-variates with $key" for (key, func) in
    Dict("rand(...)" => [rand, rand],
         "rand(rng, ...)" => [dist -> rand(rng, dist), (dist, n) -> rand(rng, dist, n)])

    for d in [W,IW]
        local d
        @test size(d) == size(func[1](d))
        @test length(d) == length(func[1](d))
        @test typeof(d)(params(d)...) == d
        @test partype(d) == Float64
    end

    @test partype(Wishart(7, Matrix{Float32}(I, 2, 2))) == Float32
    @test partype(InverseWishart(7, Matrix{Float32}(I, 2, 2))) == Float32

    @test isapprox(mean(func[2](W,100000)) , mean(W) , atol=0.1)
    @test isapprox(mean(func[2](IW,100000)), mean(IW), atol=0.1)

    v = 3.0

    @test isapprox(pdf(Wishart(v,S), S)         , 0.04507168, atol=1e-8)
    @test isapprox(pdf(Wishart(v,S), inv(S))    , 0.01327698, atol=1e-8)
    @test isapprox(pdf(Wishart(v,inv(S)),S)     , 0.0148086 , atol=1e-8)
    @test isapprox(pdf(Wishart(v,inv(S)),inv(S)), 0.01901462, atol=1e-8)

    @test isapprox(pdf(Wishart(v,S), [S, S])               , [0.04507168, 0.04507168], atol=1e-6)
    @test isapprox(pdf(Wishart(v,S), [inv(S), inv(S)])     , [0.01327698, 0.01327698], atol=1e-6)
    @test isapprox(pdf(Wishart(v,inv(S)), [S, S])          , [0.0148086, 0.0148086]  , atol=1e-6)
    @test isapprox(pdf(Wishart(v,inv(S)), [inv(S), inv(S)]), [0.01901462, 0.01901462], atol=1e-6)

    @test logpdf(Wishart(v,S), S)           ≈ log.(pdf(Wishart(v,S), S))
    @test logpdf(Wishart(v,S), inv(S))      ≈ log.(pdf(Wishart(v,S), inv(S)))
    @test logpdf(Wishart(v,inv(S)), S)      ≈ log.(pdf(Wishart(v,inv(S)), S))
    @test logpdf(Wishart(v,inv(S)), inv(S)) ≈ log.(pdf(Wishart(v,inv(S)), inv(S)))

    @test logpdf(Wishart(v,S), [S, S])                ≈ log.(pdf(Wishart(v,S), [S, S]))
    @test logpdf(Wishart(v,S), [inv(S), inv(S)])      ≈ log.(pdf(Wishart(v,S), [inv(S), inv(S)]))
    @test logpdf(Wishart(v,inv(S)), [S, S])           ≈ log.(pdf(Wishart(v,inv(S)), [S, S]))
    @test logpdf(Wishart(v,inv(S)), [inv(S), inv(S)]) ≈ log.(pdf(Wishart(v,inv(S)), [inv(S), inv(S)]))

    @test isapprox(pdf(InverseWishart(v,S), S)         , 0.04507168 , atol=1e-8)
    @test isapprox(pdf(InverseWishart(v,S), inv(S))    , 0.006247377, atol=1e-8)
    @test isapprox(pdf(InverseWishart(v,inv(S)),S)     , 0.03147137 , atol=1e-8)
    @test isapprox(pdf(InverseWishart(v,inv(S)),inv(S)), 0.01901462 , atol=1e-8)

    @test isapprox(pdf(InverseWishart(v,S), [S, S])               , [0.04507168,  0.04507168] , atol=1e-6)
    @test isapprox(pdf(InverseWishart(v,S), [inv(S), inv(S)])     , [0.006247377, 0.006247377], atol=1e-6)
    @test isapprox(pdf(InverseWishart(v,inv(S)), [S, S])          , [0.03147137,  0.03147137] , atol=1e-6)
    @test isapprox(pdf(InverseWishart(v,inv(S)), [inv(S), inv(S)]), [0.01901462,  0.01901462] , atol=1e-6)

    @test logpdf(InverseWishart(v,S), S)           ≈ log(pdf(InverseWishart(v,S), S))
    @test logpdf(InverseWishart(v,S), inv(S))      ≈ log(pdf(InverseWishart(v,S), inv(S)))
    @test logpdf(InverseWishart(v,inv(S)), S)      ≈ log(pdf(InverseWishart(v,inv(S)), S))
    @test logpdf(InverseWishart(v,inv(S)), inv(S)) ≈ log(pdf(InverseWishart(v,inv(S)), inv(S)))

    @test logpdf(InverseWishart(v,S), [S, S])                ≈ log.(pdf(InverseWishart(v,S), [S, S]))
    @test logpdf(InverseWishart(v,S), [inv(S), inv(S)])      ≈ log.(pdf(InverseWishart(v,S), [inv(S), inv(S)]))
    @test logpdf(InverseWishart(v,inv(S)), [S, S])           ≈ log.(pdf(InverseWishart(v,inv(S)), [S, S]))
    @test logpdf(InverseWishart(v,inv(S)), [inv(S), inv(S)]) ≈ log.(pdf(InverseWishart(v,inv(S)), [inv(S), inv(S)]))
end

@testset "Testing matrix-variates with $key" for (key, func) in
    Dict("rand!(..., false)" => [(dist, X) -> rand!(dist, X, false), rand!],
         "rand!(rng, ..., false)" => [(dist, X) -> rand!(rng, dist, X, false), (dist, X) -> rand!(rng, dist, X)])

    for d in [W,IW]
        local d
        local m = Matrix{Float64}(undef, size(d))
        local x = func[2](d, m)
        @test x ≡ m
        @test size(d) == size(x)
        @test length(d) == length(func[2](d, m))
        @test typeof(d)(params(d)...) == d
        @test partype(d) == Float64
    end

    m = [Matrix{partype(W)}(undef, size(W)) for _ in Base.OneTo(100000)]
    x = func[1](W,m)
    @test x ≡ m
    @test isapprox(mean(x) , mean(W) , atol=0.1)
    x = func[1](IW,m)
    @test x ≡ m
    @test isapprox(mean(x), mean(IW), atol=0.1)

    m3 = Array{partype(W), 3}(undef, size(W)..., 100000)
    x = func[2](W,m3)
    @test x ≡ m3
    @test isapprox(mean(x) , mean(mean(W)) , atol=0.1)
    x = func[2](IW,m3)
    @test x ≡ m3
    @test isapprox(mean(x), mean(mean(IW)), atol=0.1)
end

@testset "Testing matrix-variates with $key" for (key, func) in
    Dict("rand!(..., true)" => (dist, X) -> rand!(dist, X, true),
         "rand!(rng, true)" => (dist, X) -> rand!(rng, dist, X, true))

    m = Vector{Matrix{partype(W)}}(undef, 100000)
    x = func(W,m)
    @test x ≡ m
    @test isapprox(mean(x), mean(W) , atol=0.1)
    x = func(IW,m)
    @test x ≡ m
    @test isapprox(mean(x), mean(IW), atol=0.1)

end

repeats = 10
m = Vector{Matrix{partype(W)}}(undef, repeats)
rand!(W, m)
@test isassigned(m, 1)
m1=m[1]
rand!(W, m)
@test m1 ≡ m[1]
rand!(W, m, true)
@test m1 ≢ m[1]
m1 = m[1]
rand!(W, m, false)
@test m1 ≡ m[1]


