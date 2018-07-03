using Distributions
using Test, LinearAlgebra


v = 7.0
S = Matrix(1.0I, 2, 2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(v,S)
IW = InverseWishart(v,S)

for d in [W,IW]
    local d
    @test size(d) == size(rand(d))
    @test length(d) == length(rand(d))
    @test typeof(d)(params(d)...) == d
    @test partype(d) == Float64
end

@test partype(Wishart(7, Matrix{Float32}(I, 2, 2))) == Float32
@test partype(InverseWishart(7, Matrix{Float32}(I, 2, 2))) == Float32

@test isapprox(mean(rand(W,100000)) , mean(W) , atol=0.1)
@test isapprox(mean(rand(IW,100000)), mean(IW), atol=0.1)

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
