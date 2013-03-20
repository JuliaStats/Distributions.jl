using Distributions
using Test

v = 3.0
S = eye(2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(v,S)
IW = InverseWishart(v+1,S)

## these currently fail because they're not quite approx enough
@test_approx_eq mean(rand(W,1000000)) mean(W)
@test_approx_eq mean(rand(IW,1000000)) mean(IW)

@test_approx_eq pdf(Wishart(v,S), S) 0.04507168
@test_approx_eq pdf(Wishart(v,S), inv(S)) 0.01327698
@test_approx_eq pdf(Wishart(v,inv(S)),S) 0.0148086
@test_approx_eq pdf(Wishart(v,inv(S)),inv(S)) 0.01901462

@test_approx_eq pdf(InverseWishart(v,S), S) 0.04507168
@test_approx_eq pdf(InverseWishart(v,S), inv(S)) 0.006247377
@test_approx_eq pdf(InverseWishart(v,inv(S)),S)  0.03147137
@test_approx_eq pdf(InverseWishart(v,inv(S)),inv(S)) 0.01901462

