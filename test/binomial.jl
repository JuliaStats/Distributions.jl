using Distributions
using Test, Random


# Test the consistency between the recursive and nonrecursive computation of the pdf
# of the Binomial distribution
Random.seed!(1234)
for (p, n) in [(0.6, 10), (0.8, 6), (0.5, 40), (0.04, 20), (1., 100), (0., 10), (0.999999, 1000), (1e-7, 1000)]
    local p

    d = Binomial(n, p)

    a = pdf.(d, 0:n)
    for t=0:n
        @test pdf(d, t) ≈ a[1+t]
    end

    li = rand(0:n, 2)
    rng = minimum(li):maximum(li)
    b = pdf.(d, rng)
    for t in rng
        @test pdf(d, t) ≈ b[t - first(rng) + 1]
    end

end

# Test calculation of expectation value for Binomial distribution
@test Distributions.expectation(Binomial(6), identity) ≈ 3.0
@test Distributions.expectation(Binomial(10, 0.2), x->-x) ≈ -2.0

# Test mode
@test Distributions.mode(Binomial(100, 0.4)) == 40
@test Distributions.mode(Binomial(1, 0.51)) == 1
@test Distributions.mode(Binomial(1, 0.49)) == 0
