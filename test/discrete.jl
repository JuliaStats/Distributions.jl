#  Unit testing of (bounded) univariate discrete distributions
#
#  Here, bounded means the sample values are bounded. 
#
#  Distributions covered by this suite:
#
# 	- Bernoulli
#  	- Categorical
#   - DiscreteUniform
#

import NumericExtensions
using Distributions
using Base.Test


for d in [
    Bernoulli(0.1),
    Bernoulli(0.5),
    Bernoulli(0.9), 
    Categorical([0.1, 0.9]),
    Categorical([0.5, 0.5]),
    Categorical([0.9, 0.1]), 
    Categorical([0.2, 0.5, 0.3]), 
    DiscreteUniform(0, 3),
    DiscreteUniform(2.0, 5.0),
    Binomial(1, 0.5),
    Binomial(100, 0.1),
    Binomial(100, 0.9),
    Binomial(10000, 0.03)]

    # println(d)

    xmin = min(d)
    xmax = max(d)
    @assert isa(xmin, Int)
    @assert isa(xmax, Int)
    @assert xmin <= xmax

    ####
    #
    #  Part 1:  testing the capability of sampling
    #  
    ####

    n = 10000

    # check that we can generate a single random draw
    draw = rand(d)
    @test isa(draw, Int)
    @test xmin <= draw <= xmax

    # check that we can generate many random draws at once
    x = rand(d, n)
    @test isa(x, Vector{Int})
    @test xmin <= min(x) <= max(x) <= xmax

    # check that we can generate many random draws in-place
    rand!(d, x)
    @test xmin <= min(x) <= max(x) <= xmax

    ####
    #
    #  Part 2: testing insupport
    #  
    ####

    x = [xmin:xmax]
    n = length(x)

    @test !insupport(d, xmin-1)
    @test !insupport(d, xmax+1)
    for i = 1:n
        @test insupport(d, x[i])
    end
    @test insupport(d, x)

    ####
    #
    #  Part 3: testing evaluation
    #  
    ####

    p = Array(Float64, n)
    c = Array(Float64, n)
    cc = Array(Float64, n)

    lp = Array(Float64, n)
    lc = Array(Float64, n)
    lcc = Array(Float64, n)

    ci = 0.

    for i in 1 : n
        p[i] = pdf(d, x[i])        
        ci += p[i]

        c[i] = cdf(d, x[i])
        cc[i] = ccdf(d, x[i])

        @test_approx_eq ci c[i]
        @test_approx_eq c[i] + cc[i] 1.0

        lp[i] = logpdf(d, x[i])
        lc[i] = logcdf(d, x[i])
        lcc[i] = logccdf(d, x[i])

        @test_approx_eq_eps exp(lp[i]) p[i] 1.0e-12
        @test_approx_eq_eps exp(lc[i]) c[i] 1.0e-12
        @test_approx_eq_eps exp(lcc[i]) cc[i] 1.0e-12

        if !isa(d, Binomial)
            @test quantile(d, c[i] - 1.0e-8) == x[i]
            @test cquantile(d, cc[i] + 1.0e-8) == x[i]
            @test invlogcdf(d, lc[i] - 1.0e-8) == x[i]

            if 0.0 < c[i] < 1.0
                @test invlogccdf(d, lcc[i] + 1.0e-8) == x[i]
            end
        end
    end

    # check consistency of scalar-based and vectorized evaluation

    @test_approx_eq pdf(d, x) p
    @test_approx_eq cdf(d, x) c
    @test_approx_eq ccdf(d, x) cc

    @test_approx_eq logpdf(d, x) lp
    @test_approx_eq logcdf(d, x) lc
    @test_approx_eq logccdf(d, x) lcc

    ####
    #
    #  Part 4: testing statistics
    #  
    ####    

    xf = float64(x)
    xmean = dot(p, xf)
    xvar = dot(p, abs2(xf - xmean))
    xstd = sqrt(xvar)
    xentropy = NumericExtensions.entropy(p)
    xskew = dot(p, (xf - xmean).^3) / (xstd.^3)
    xkurt = dot(p, (xf - xmean).^4) / (xvar.^2) - 3.0

    @test_approx_eq mean(d)     xmean
    @test_approx_eq var(d)      xvar
    @test_approx_eq std(d)      xstd
    @test_approx_eq skewness(d) xskew
    @test_approx_eq kurtosis(d) xkurt
    @test_approx_eq entropy(d)  xentropy

end


