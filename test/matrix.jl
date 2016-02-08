using Distributions
using Base.Test

v = 7.0
S = eye(2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(v,S)
IW = InverseWishart(v,S)

for d in [W,IW]
    @test size(d) == size(rand(d))
    @test length(d) == length(rand(d))
    @test typeof(d)(params(d)...) == d
end

@test_approx_eq_eps mean(rand(W,100000)) mean(W) 0.1
@test_approx_eq_eps mean(rand(IW,100000)) mean(IW) 0.1

v = 3.0

@test_approx_eq_eps pdf(Wishart(v,S), S) 0.04507168 1e-8
@test_approx_eq_eps pdf(Wishart(v,S), inv(S)) 0.01327698 1e-8
@test_approx_eq_eps pdf(Wishart(v,inv(S)),S) 0.0148086 1e-8
@test_approx_eq_eps pdf(Wishart(v,inv(S)),inv(S)) 0.01901462 1e-8

@test_approx_eq_eps pdf(InverseWishart(v,S), S) 0.04507168 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,S), inv(S)) 0.006247377 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,inv(S)),S)  0.03147137 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,inv(S)),inv(S)) 0.01901462 1e-8
