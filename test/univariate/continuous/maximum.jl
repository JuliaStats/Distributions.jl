using Test, Distributions
using QuadGK: quadgk

for n in 10:10:100
    Hn = sum(x->1/x, 1:n)
    Mn = Maximum(Exponential(), n)
    # the expectation of the maximum of n exponential random variables is H(n) = 1/1 + 1/2 + ... + 1/n
    @test quadgk(x -> pdf(Mn, x)*x, 0, 1000)[1] ≈ H(n)

    # the variance of the maximum of n exponential random variables converges to π^2/6
    @test abs(quadgk(x -> pdf(Maximum(Exponential(), n), x)*(x-Hn)^2, 0, 1000)[1] - π^2/6) < 1/n
end