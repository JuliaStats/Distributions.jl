using Distributions
using Test, Random

Random.seed!(1234)

@testset "binomial" begin
# Test the consistency between the recursive and nonrecursive computation of the pdf
# of the Binomial distribution
for (p, n) in [(0.6, 10), (0.8, 6), (0.5, 40), (0.04, 20), (1., 100), (0., 10), (0.999999, 1000), (1e-7, 1000)]
    d = Binomial(n, p)

    a = Base.Fix1(pdf, d).(0:n)
    for t=0:n
        @test pdf(d, t) ≈ a[1+t]
    end

    li = rand(0:n, 2)
    rng = minimum(li):maximum(li)
    b = Base.Fix1(pdf, d).(rng)
    for t in rng
        @test pdf(d, t) ≈ b[t - first(rng) + 1]
    end
end

# Test calculation of expectation value for Binomial distribution
@test Distributions.expectation(identity, Binomial(6)) ≈ 3.0
@test Distributions.expectation(x -> -x, Binomial(10, 0.2)) ≈ -2.0

# Test median
@test median(Binomial(25,3//10)) == 7
@test median(Binomial(45,3//10)) == 13
@test median(Binomial(65,3//10)) == 19
@test median(Binomial(85,3//10)) == 25

@test all(median(Binomial(7, p)) == quantile(Binomial(7, p), 1//2) for p in 0:0.1:1)
    
# Test mode
@test mode(Binomial(100, 0.4)) == 40
@test mode(Binomial(1, 0.51)) == 1
@test mode(Binomial(1, 0.49)) == 0
@test mode(Binomial(4, 2//3)) == 3
@test mode(Binomial(6, 2//7)) == 1
@test mode(Binomial(7, 1//8)) == 0

@test modes(Binomial(4, 2//3)) == [3]
@test modes(Binomial(5, 3//4)) == [4]
@test modes(Binomial(3, 2//4)) == [1, 2]
@test modes(Binomial(4, 2//5)) == [1, 2]
@test modes(Binomial(6, 2//7)) == [1, 2]
@test modes(Binomial(6, 3//7)) == [2, 3]
@test modes(Binomial(7, 1//8)) == [0, 1]

@test isplatykurtic(Bernoulli(0.5))
@test ismesokurtic(Normal(0.0, 1.0))
@test isleptokurtic(Laplace(0.0, 1.0))
end
