## chernoff.jl
##
## The code below is intended to go with the Distributions package of Julia.
## It was written by Joris Pinkse, joris@psu.edu, on December 12, 2017.  Caveat emptor.
##
## It computes pdf, cdf, moments, quantiles, and random numbers for the Chernoff distribution.
##
## Most of the symbols have the same meaning as in the Groeneboom and Wellner paper.
##
## Random numbers are drawn using a Ziggurat algorithm.  To obtain draws in the tails, the
## algorithm reverts to quantiles, which is slow.
"""
    Chernoff()

The *Chernoff distribution* is the distribution of the random variable
```math
\\underset{t \\in (-\\infty,\\infty)}{\\arg\\max} ( G(t) - t^2 ),
```
where ``G`` is standard two-sided Brownian motion.

The distribution arises as the limit distribution of various cube-root-n consistent estimators,
including the isotonic regression estimator of Brunk, the isotonic density estimator of Grenander,
the least median of squares estimator of Rousseeuw, and the maximum score estimator of Manski.

For theoretical results, see e.g. Kim and Pollard, Annals of Statistics, 1990.  The code for the
computation of pdf and cdf is based on the algorithm described in Groeneboom and Wellner,
Journal of Computational and Graphical Statistics, 2001.

```julia
cdf(Chernoff(),-x)              # For tail probabilities, use this instead of 1-cdf(Chernoff(),x)
```
"""
struct Chernoff <: ContinuousUnivariateDistribution
end

module ChernoffComputations
    import QuadGK.quadgk
    # The following arrays of constants have been precomputed to speed up computation.
    # The first two correspond to the arrays a and b in the Groeneboom and Wellner article.
    # The array airyai_roots contains roots of the airyai functions (atilde in the paper).
    # Finally airyai_prime contains contains airyaiprime evaluated at the corresponding atildes
    const a=[
         0.14583333333333334
        -0.0014105902777777765
        -2.045269786155239e-5
         1.7621771897199738e-6
        -5.7227597511834636e-8
         1.2738111577329235e-9
        -2.1640824986157608e-11
         2.87925759635786e-13
        -2.904496582565578e-15
         1.8213982724416428e-17
         4.318809086319947e-20
        -3.459634830348895e-21
         6.541295360751684e-23
        -8.92275524974091e-25
         1.0102911018761945e-26
        -9.976759882474874e-29
        ]
    const b=[
         0.6666666666666666
         0.021164021164021163
        -0.0007893341226674574
         1.9520978781876193e-5
        -3.184888134149933e-7
         2.4964953625552273e-9
         3.6216439304006735e-11
        -1.874438299727797e-12
         4.393984966701305e-14
        -7.57237085924526e-16
         1.063018673013022e-17
        -1.26451831871265e-19
         1.2954000902764143e-21
        -1.1437790369170514e-23
         8.543413293195206e-26
        -5.05879944592705e-28
        ]
    const airyai_roots=[
        -2.338107410459767
        -4.08794944413097
        -5.520559828095551
        -6.786708090071759
        -7.944133587120853
        -9.02265085334098
        -10.040174341558085
        -11.008524303733262
        -11.936015563236262
        -12.828776752865757
        -13.691489035210719
        -14.527829951775335
        -15.340755135977997
        -16.132685156945772
        -16.90563399742994
        -17.66130010569706
        ]
    const airyai_prime=[
         0.7012108227206906
        -0.8031113696548534
         0.865204025894141
        -0.9108507370495821
         0.9473357094415571
        -0.9779228085695176
         1.0043701226603126
        -1.0277386888207862
         1.0487206485881893
        -1.0677938591574279
         1.0853028313507
        -1.1015045702774968
         1.1165961779326556
        -1.1307323104931881
         1.1440366732735523
        -1.156609849116565
        ]

    const cuberoottwo = cbrt(2.0)
    const sqrthalfpi = sqrt(0.5*pi)
    const sqrttwopi = sqrt(2.0*pi)

    function p(y::Real)
        if iszero(y)
            return -sqrt(0.5*pi)
        end

        (y > 0) || throw(DomainError(y, "argument must be positive"))

        cnsty = y^(-1.5)
        if (y <= 1.0)
            return sum([(b[k]*cnsty - a[k]*sqrthalfpi)*y^(3*k) for k=1:length(a)])-sqrthalfpi
        else
            return sum([exp(cuberoottwo*a*y) for a in airyai_roots]) * 2 * sqrttwopi * exp(-y*y*y/6) - cnsty
        end
    end

    function g(x::Real)
        function g_one(y::Real)
            return p(y) * exp(-0.5*y*(2*x+y)*(2*x+y))
        end
        function g_two(y::Real)
            z = 2*x+y*y
            return (z*y*y + 0.5 * z*z) * exp(-0.5*y*y*z*z)
        end
        if (x <= -1.0)
            return cuberoottwo*cuberoottwo * exp(2*x*x*x/3.0) * sum([exp(-cuberoottwo*airyai_roots[k]*x) / airyai_prime[k] for k=1:length(airyai_roots)])
        else
            return 2*x - (quadgk(g_one, 0.0, Inf)[1] - 4*quadgk(g_two, 0.0, Inf)[1]) / sqrttwopi   # should perhaps combine integrals
        end
    end

    _pdf(x::Real) = g(x)*g(-x)*0.5
    _cdf(x::Real) = (x < 0.0) ? _cdfbar(-x) : 0.5 + quadgk(_pdf,0.0,x)[1]
    _cdfbar(x::Real) = (x < 0.0) ? _cdf(x) : quadgk(_pdf, x, Inf)[1]
end

pdf(d::Chernoff, x::Real) = ChernoffComputations._pdf(x)
logpdf(d::Chernoff, x::Real) = log(ChernoffComputations.g(x))+log(ChernoffComputations.g(-x))+log(0.5)
cdf(d::Chernoff, x::Real) = ChernoffComputations._cdf(x)

function quantile(d::Chernoff, tau::Real)
    # some commonly used quantiles were precomputed
    precomputedquants=[
        0.0 -Inf;
        0.01 -1.171534341573129;
        0.025 -0.9981810946684274;
        0.05 -0.8450811886357725;
        0.1 -0.6642351964332931;
        0.2 -0.43982766604886553;
        0.25 -0.353308035220429;
        0.3 -0.2751512847290148;
        0.4 -0.13319637678583637;
        0.5 0.0;
        0.6 0.13319637678583637;
        0.7 0.2751512847290147;
        0.75 0.353308035220429;
        0.8 0.4398276660488655;
        0.9 0.6642351964332931;
        0.95 0.8450811886357724;
        0.975 0.9981810946684272;
        0.99 1.17153434157313;
        1.0 Inf
        ]
    (0.0 <= tau && tau <= 1.0) || throw(DomainError(tau, "illegal value of tau"))
    present = searchsortedfirst(precomputedquants[:, 1], tau)
    if present <= size(precomputedquants, 1)
        if tau == precomputedquants[present, 1]
            return precomputedquants[present, 2]
        end
    end

    # one good approximation of the quantiles can be computed using Normal(0.0, stdapprox) with stdapprox = 0.52
    stdapprox = 0.52
    dnorm = Normal(0.0, 1.0)
    if tau < 0.001
        return -newton(x -> tau - ChernoffComputations._cdfbar(x), ChernoffComputations._pdf, quantile(dnorm, 1.0 - tau)*stdapprox)

    end
    if tau > 0.999
        return newton(x -> 1.0 - tau - ChernoffComputations._cdfbar(x), ChernoffComputations._pdf, quantile(dnorm, tau)*stdapprox)
    end
    return newton(x -> ChernoffComputations._cdf(x) - tau, ChernoffComputations._pdf, quantile(dnorm, tau)*stdapprox)   # should consider replacing x-> construct for speed
end

minimum(d::Chernoff) = -Inf
maximum(d::Chernoff) = Inf
insupport(d::Chernoff, x::Real) = isnan(x) ? false : true

mean(d::Chernoff) = 0.0
var(d::Chernoff) = 0.26355964132470455
modes(d::Chernoff) = [0.0]
mode(d::Chernoff) = 0.0
skewness(d::Chernoff) = 0.0
kurtosis(d::Chernoff) = -0.16172525511461888
kurtosis(d::Chernoff, excess::Bool) = kurtosis(d) + (excess ? 0.0 : 3.0)
entropy(d::Chernoff) = -0.7515605300273104

### Random number generation
rand(d::Chernoff) = rand(default_rng(), d)
function rand(rng::AbstractRNG, d::Chernoff)                 # Ziggurat random number generator --- slow in the tails
    # constants needed for the Ziggurat algorithm
    A = 0.03248227216266608
    x = [
        1.4765521793744492
        1.3583996502410562
        1.2788224934376338
        1.2167121025431031
        1.164660153310361
        1.1191528874523227
        1.0782281238946987
        1.0406685077375248
        1.0056599129954287
        0.9726255909850547
        0.9411372703351518
        0.910863725882819
        0.8815390956471935
        0.8529422519848634
        0.8248826278765808
        0.7971898990526088
        0.769705944382039
        0.7422780374708061
        0.7147524811697309
        0.6869679724643997
        0.6587479056872714
        0.6298905501492661
        0.6001554584431008
        0.5692432986584453
        0.5367639126935895
        0.5021821750911241
        0.4647187417226889
        0.42314920361072667
        0.37533885097957154
        0.31692143952775814
        0.2358977457249061
        1.0218214689661219e-7
       ]
    y=[
        0.02016386420423385
        0.042162593823411566
        0.06607475557706186
        0.09147489698007219
        0.11817165794330549
        0.14606157249935905
        0.17508555351826158
        0.20521115629454248
        0.2364240467790298
        0.26872350680813245
        0.3021199879039766
        0.3366338395880132
        0.37229479658922043
        0.4091420247190392
        0.4472246406136998
        0.48660269400571066
        0.5273486595545326
        0.5695495447104147
        0.6133097942704644
        0.6587552778727138
        0.7060388099508409
        0.7553479187650504
        0.8069160398536568
        0.8610391369707446
        0.9181013320624011
        0.97861633948841
        1.0432985878313983
        1.113195213997703
        1.1899583790598025
        1.2764995726449935
        1.3789927085182485
        1.516689116183566
        ]
    n = length(x)
    i = rand(rng, 0:n-1)
    r = (2.0*rand(rng)-1) * ((i>0) ? x[i] : A/y[1])
    rabs = abs(r)
    if rabs < x[i+1]
        return r
    end
    s = (i>0) ? (y[i]+rand(rng)*(y[i+1]-y[i])) : rand(rng)*y[1]
    if s < 2.0*ChernoffComputations._pdf(rabs)
        return r
    end
    if i > 0
        return rand(rng, d)
    end
    F0 = ChernoffComputations._cdf(A/y[1])
    tau = 2.0*rand(rng)-1 # ~ U(-1,1)
    tauabs = abs(tau)
    return quantile(d, tauabs + (1-tauabs)*F0) * sign(tau)
end
