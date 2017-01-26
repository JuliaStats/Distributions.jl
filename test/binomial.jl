using Distributions
using Base.Test

# Test the consistency between the recursive and nonrecursive computation of the pdf 
# of the Binomial distribution
srand(1234)
for (p, n) in [(0.6, 10), (0.8, 6), (0.5, 40), (0.04, 20), (1., 100), (0., 10), (0.999999, 1000), (1e-7, 1000)]

    d = Binomial(n, p)

    a = pdf(d, 0:n)
    for t=0:n
        @test pdf(d, t) ≈ a[1+t]
    end

    li = rand(0:n, 2)
    rng = minimum(li):maximum(li)
    b = pdf(d, rng)
    for t in rng
        @test pdf(d, t) ≈ b[t - first(rng) + 1]
    end

end
