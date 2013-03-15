using Distributions
using Test

nu = 3.0
S = eye(2)
S[1, 2] = S[2, 1] = 0.5

W = Wishart(nu,S)

Schol = cholfact(S,'U')
W2 = Wishart(nu, Schol)

nu = convert(Int64, nu)
W3 = Wishart(nu, S)
W4 = Wishart(nu, Schol)

##@test_approx_eq mean(rand(W,1000000)) mean(W)

IW = InverseWishart(nu+1, S)

##@test_approx_eq pdf(W, S) 0.04507168
##@test_approx_eq pdf(IW, S) 0.02253584