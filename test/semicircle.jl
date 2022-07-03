using Distributions
using Random: MersenneTwister
using  Test

d = Semicircle(2.0)

@test params(d) == (2.0,)

@test minimum(d) == -2.0
@test maximum(d) == +2.0
@test extrema(d) == (-2.0, 2.0)

@test mean(d)     ==  .0
@test var(d)      == 1.0
@test skewness(d) ==  .0
@test median(d)   ==  .0
@test mode(d)     ==  .0
@test entropy(d)  == 1.33787706640934544458

@test pdf(d, -5) == .0
@test pdf(d, -2) == .0
@test pdf(d,  0) == .31830988618379067154
@test pdf(d, +2) == .0
@test pdf(d, +5) == .0

@test logpdf(d, -5) == -Inf
@test logpdf(d, -2) == -Inf
@test logpdf(d,  0) ≈  log(.31830988618379067154)
@test logpdf(d, +2) == -Inf
@test logpdf(d, +5) == -Inf

@test cdf(d, -5) ==  .0
@test cdf(d, -2) ==  .0
@test cdf(d,  0) ==  .5
@test cdf(d, +2) == 1.0
@test cdf(d, +5) == 1.0

@test quantile(d,  .0) == -2.0
@test quantile(d,  .5) ==   .0
@test quantile(d, 1.0) == +2.0

function test_sample_uniform(rng,sample)
    N = length(sample)
    sample = sort(sample)
    ks = rand(rng,eachindex(sample), 20)
    ifirst = firstindex(sample)
    ilast = lastindex(sample)
    imid = round(Int,(ifirst + ilast) / 2)
    push!(ks, ifirst)
    push!(ks, ilast)
    push!(ks, imid)
    for k in ks
        # test k-th order statistic of sample is sane
        dk = Beta(k, N+1-k)
        lo = quantile(dk, 0.001)
        hi = quantile(dk, 0.999)
        x = sample[k]
        @test lo < x < hi
    end
    # test sample mean compatible with central limit theorem
    dmean = Normal(0.5, std(Uniform(0,1))/√(N))
    @test quantile(dmean, 0.01) < mean(sample) < quantile(dmean, 0.99)
end

N = 10^3
rng = MersenneTwister(0)
for r in rand(rng, Uniform(0,10), 5)
    semi = Semicircle(r)
    sample = rand(rng, semi, N) 
    mi, ma = extrema(sample)
    @test -r <= mi < ma <= r
    test_sample_uniform(rng,cdf.(semi, sample))
end
