using Distributions
using Test

# Test the special base where PoissonBinomial distribution reduces
# to Binomial distribution
for (p, n) in [(0.8, 6), (0.5, 10), (0.04, 20)]
    local p

    d = PoissonBinomial(fill(p, n))
    dref = Binomial(n, p)
    println("   testing PoissonBinomial p=$p, n=$n")

    @test isa(d, PoissonBinomial)
    @test minimum(d) == 0
    @test maximum(d) == n
    @test extrema(d) == (0, n)
    @test ntrials(d) == n
    @test entropy(d)  ≈ entropy(dref)
    @test median(d)   ≈ median(dref)
    @test mean(d)     ≈ mean(dref)
    @test var(d)      ≈ var(dref)
    @test kurtosis(d) ≈ kurtosis(dref)
    @test skewness(d) ≈ skewness(dref)

    for t=0:5
        @test mgf(d, t) ≈ mgf(dref, t)
        @test cf(d, t)  ≈ cf(dref, t)
    end
    for i=0.1:0.1:.9
        @test quantile(d, i) ≈ quantile(dref, i)
    end
    for i=0:n
        @test isapprox(cdf(d, i), cdf(dref, i), atol=1e-15)
        @test isapprox(pdf(d, i), pdf(dref, i), atol=1e-15)
    end

end

# Test against a sum of three Binomial distributions
for (n₁, n₂, n₃, p₁, p₂, p₃) in [(10, 10, 10, 0.1, 0.5, 0.9),
                                 (1, 10, 100, 0.99, 0.1, 0.05),
                                 (5, 1, 3, 0.01, 0.99, 0.999),
                                 (10, 7, 10, 0., 0.9, 0.5)]

    n = n₁ + n₂ + n₃
    p = zeros(n)
    p[1:n₁] .= p₁
    p[n₁+1: n₁ + n₂] .= p₂
    p[n₁ + n₂ + 1:end] .= p₃
    d = PoissonBinomial(p)
    println("   testing PoissonBinomial [$(n₁) × $(p₁), $(n₂) × $(p₂), $(n₃) × $(p₃)]")
    b1 = Binomial(n₁, p₁)
    b2 = Binomial(n₂, p₂)
    b3 = Binomial(n₃, p₃)

    pmf1 = pdf.(b1, support(b1))
    pmf2 = pdf.(b2, support(b2))
    pmf3 = pdf.(b3, support(b3))

    @test mean(d) ≈ (mean(b1) + mean(b2) + mean(b3))
    @test var(d)  ≈ (var(b1) + var(b2) + var(b3))
    for t=0:5
        @test mgf(d, t) ≈ (mgf(b1, t) * mgf(b2, t) * mgf(b3, t))
        @test cf(d, t)  ≈ (cf(b1, t) * cf(b2, t) * cf(b3, t))
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
        @test isapprox(pdf(d, k), m, atol=1e-15)
    end
end

# Test the _dft helper function
@testset "_dft" begin
    x = Distributions._dft(collect(1:8))
    # Comparator computed from FFTW
    fftw_fft = [36.0 + 0.0im,
                -4.0 + 9.65685424949238im,
                -4.0 + 4.0im,
                -4.0 + 1.6568542494923806im,
                -4.0 + 0.0im,
                -4.0 - 1.6568542494923806im,
                -4.0 - 4.0im,
                -4.0 - 9.65685424949238im]
    @test x ≈ fftw_fft
end

# Test autodiff using ForwardDiff
f = x -> logpdf(PoissonBinomial(x), 0)
at = [0.5, 0.5]
@test isapprox(ForwardDiff.gradient(f, at), fdm(f, at), atol=1e-6)
