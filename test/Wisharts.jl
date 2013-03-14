using Distributions
using Test

nu = 3.0
S = eye(2)
S[1,2] = S[2,1] = 0.5

W = Wishart(nu,S)

Schol = cholfact(S,'U')
W2 = Wishart(nu, Schol)

nu = convert(Int64, nu)
W3 = Wishart(nu, S)
W4 = Wishart(nu, Schol)

##@test_approx_eq mean(rand(W,1000000)) mean(W)

IW = InverseWishart(nu, S)

