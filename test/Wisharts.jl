using Distributions
using Test

v = 3.0
S = eye(2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(v,S)
IW = InverseWishart(v+1,S)

@test_approx_eq_eps mean(rand(W,100000)) mean(W) 1e-1
@test_approx_eq_eps mean(rand(IW,100000)) mean(IW) 1e-1

@test_approx_eq_eps pdf(Wishart(v,S), S) 0.04507168 1e-8
@test_approx_eq_eps pdf(Wishart(v,S), inv(S)) 0.01327698 1e-8
@test_approx_eq_eps pdf(Wishart(v,inv(S)),S) 0.0148086 1e-8
@test_approx_eq_eps pdf(Wishart(v,inv(S)),inv(S)) 0.01901462 1e-8

@test_approx_eq_eps pdf(InverseWishart(v,S), S) 0.04507168 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,S), inv(S)) 0.006247377 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,inv(S)),S)  0.03147137 1e-8
@test_approx_eq_eps pdf(InverseWishart(v,inv(S)),inv(S)) 0.01901462 1e-8

