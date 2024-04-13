using Distributions
using ChainRulesTestUtils
using ForwardDiff
using Test

@testset "poissonbinomial" begin
function naive_esf(x::AbstractVector{T}) where T <: Real
    n = length(x)
    S = zeros(T, n+1)
    states = hcat(reverse.(digits.(0:2^n-1,base=2,pad=n))...)'

    r_states = vec(mapslices(sum, states, dims=2))

    for r in 0:n
        idx = findall(r_states .== r)
        S[r+1] = sum(mapslices(x->prod(x[x .!= 0]), states[idx, :] .* x', dims=2))
    end
    return S
end

function naive_pb(p::AbstractVector{T}) where T <: Real
    x = p ./ (1 .- p)
    naive_esf(x) * prod(1 ./ (1 .+ x))
end

# test PoissonBinomial PMF algorithms
x = [3.5118, .6219, .2905, .8450, 1.8648]
n = length(x)
p = x ./ (1 .+ x)

naive_sol = naive_pb(p)

@test Distributions.poissonbinomial_pdf_fft(p) ≈ naive_sol
@test Distributions.poissonbinomial_pdf(p) ≈ naive_sol
@test Distributions.poissonbinomial_pdf(Tuple(p)) ≈ naive_sol

@test Distributions.poissonbinomial_pdf_fft(p) ≈ Distributions.poissonbinomial_pdf(p)

# Test the special base where PoissonBinomial distribution reduces
# to Binomial distribution
for (p, n) in [(0.8, 6), (0.5, 10), (0.04, 20)]
    d = PoissonBinomial(fill(p, n))
    dref = Binomial(n, p)
    println("   testing PoissonBinomial p=$p, n=$n")

    @test isa(d, PoissonBinomial)
    @test minimum(d) == 0
    @test maximum(d) == n
    @test extrema(d) == (0, n)
    @test ntrials(d) == n
    @test @inferred(entropy(d))  ≈ entropy(dref)
    @test @inferred(median(d))   ≈ median(dref)
    @test @inferred(mean(d))     ≈ mean(dref)
    @test @inferred(var(d))      ≈ var(dref)
    @test @inferred(kurtosis(d)) ≈ kurtosis(dref)
    @test @inferred(skewness(d)) ≈ skewness(dref)

    for t=0:5
        @test @inferred(mgf(d, t)) ≈ mgf(dref, t)
        @test @inferred(cf(d, t))  ≈ cf(dref, t)
    end
    for i=0.1:0.1:.9
        @test @inferred(quantile(d, i)) ≈ quantile(dref, i)
    end
    for i=0:n
        @test @inferred(pdf(d, i)) ≈ pdf(dref, i) atol=1e-14
        @test @inferred(pdf(d, i//1)) ≈ pdf(dref, i) atol=1e-14
        @test @inferred(logpdf(d, i)) ≈ logpdf(dref, i)
        @test @inferred(logpdf(d, i//1)) ≈ logpdf(dref, i)
        for f in (cdf, ccdf, logcdf, logccdf)
            @test @inferred(f(d, i)) ≈ f(dref, i) rtol=1e-6
            @test @inferred(f(d, i//1)) ≈ f(dref, i) rtol=1e-6
            @test @inferred(f(d, i + 0.5)) ≈ f(dref, i) rtol=1e-6
        end
    end

    @test iszero(@inferred(cdf(d, -Inf)))
    @test isone(@inferred(cdf(d, Inf)))
    @test @inferred(logcdf(d, -Inf)) == -Inf
    @test iszero(@inferred(logcdf(d, Inf)))
    @test isone(@inferred(ccdf(d, -Inf)))
    @test iszero(@inferred(ccdf(d, Inf)))
    @test iszero(@inferred(logccdf(d, -Inf)))
    @test @inferred(logccdf(d, Inf)) == -Inf
    for f in (cdf, ccdf, logcdf, logccdf)
        @test isnan(f(d, NaN))
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

    pmf1 = Base.Fix1(pdf, b1).(support(b1))
    pmf2 = Base.Fix1(pdf, b2).(support(b2))
    pmf3 = Base.Fix1(pdf, b3).(support(b3))

    @test @inferred(mean(d)) ≈ (mean(b1) + mean(b2) + mean(b3))
    @test @inferred(var(d))  ≈ (var(b1) + var(b2) + var(b3))
    for t=0:5
        @test @inferred(mgf(d, t)) ≈ (mgf(b1, t) * mgf(b2, t) * mgf(b3, t))
        @test @inferred(cf(d, t))  ≈ (cf(b1, t) * cf(b2, t) * cf(b3, t))
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
        @test @inferred(pdf(d, k)) ≈ m atol=1e-14
        @test @inferred(pdf(d, k//1)) ≈ m atol=1e-14
        @test @inferred(logpdf(d, k)) ≈ log(m)
        @test @inferred(logpdf(d, k//1)) ≈ log(m)
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

@testset "automatic differentiation" begin
    # Test autodiff using ForwardDiff
    f = x -> logpdf(PoissonBinomial(x), 0)
    at = [0.5, 0.5]
    @test isapprox(ForwardDiff.gradient(f, at), fdm(f, at), atol=1e-6)

    # Test ChainRules definition
    for f in (Distributions.poissonbinomial_pdf, Distributions.poissonbinomial_pdf_fft)
        test_frule(f, rand(50))
        test_rrule(f, rand(50))
    end
end
end
