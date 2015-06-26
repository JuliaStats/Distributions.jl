using Distributions
using Base.Test

# Test the special base where PoissonBinomial distribution reduces 
# to Binomial distribution
for (p, n) in [(0.8, 6), (0.5, 10), (0.04, 20)]

    d = PoissonBinomial(fill(p, n))
    dref = Binomial(n, p)
    println("   testing PoissonBinomial p=$p, n=$n")

    @test isa(d, PoissonBinomial)
    @test minimum(d) == 0
    @test maximum(d) == n
    @test ntrials(d) == n
    @test_approx_eq entropy(d) entropy(dref)
    @test_approx_eq median(d) median(dref)
    @test_approx_eq mean(d) mean(dref)
    @test_approx_eq var(d) var(dref)
    @test_approx_eq kurtosis(d) kurtosis(dref)
    @test_approx_eq skewness(d) skewness(dref)
    
    for t=0:5
        @test_approx_eq mgf(d, t) mgf(dref, t)
        @test_approx_eq cf(d, t) cf(dref, t)
    end
    for i=0.1:0.1:.9
        @test_approx_eq quantile(d, i) quantile(dref, i)
    end
    for i=0:n
        @test_approx_eq_eps cdf(d, i) cdf(dref, i) 1e-15
        @test_approx_eq_eps pdf(d, i) pdf(dref, i) 1e-15
    end

end

# Test against a sum of three Binomial distributions
for (n₁, n₂, n₃, p₁, p₂, p₃) in [(10, 10, 10, 0.1, 0.5, 0.9), 
                                 (1, 10, 100, 0.99, 0.1, 0.05),
                                 (5, 1, 3, 0.01, 0.99, 0.999),
                                 (10, 7, 10, 0., 0.9, 0.5)]

    n = n₁ + n₂ + n₃
    p = zeros(n)
    p[1:n₁] = p₁
    p[n₁+1: n₁ + n₂] = p₂
    p[n₁ + n₂ + 1:end] = p₃
    d = PoissonBinomial(p)
    println("   testing PoissonBinomial [$(n₁) × $(p₁), $(n₂) × $(p₂), $(n₃) × $(p₃)]")
    b1 = Binomial(n₁, p₁)
    b2 = Binomial(n₂, p₂)
    b3 = Binomial(n₃, p₃)

    pmf1 = pdf(b1)
    pmf2 = pdf(b2)
    pmf3 = pdf(b3)

    @test_approx_eq mean(d) (mean(b1) + mean(b2) + mean(b3))
    @test_approx_eq var(d) (var(b1) + var(b2) + var(b3))
    for t=0:5
        @test_approx_eq mgf(d, t) (mgf(b1, t) * mgf(b2, t) * mgf(b3, t))
        @test_approx_eq cf(d, t) (cf(b1, t) * cf(b2, t) * cf(b3, t))
    end

    for k=0:n
        m = 0.
        for i=0:min(n₁, k)
            mc = 0.
            for j=i:min(i+n₂, k)
                mc += (k - j <= n₃) && pmf2[j-i+1] * pmf3[k-j+1]
            end
            m += pmf1[i+1] * mc
        end
        @test_approx_eq_eps pdf(d, k) m 1e-15
    end
end
