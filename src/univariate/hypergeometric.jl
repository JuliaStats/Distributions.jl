immutable HyperGeometric <: DiscreteUnivariateDistribution
    ns::Int # number of successes in population
    nf::Int # number of failures in population
    n::Int  # sample size
    function HyperGeometric(s::Real, f::Real, n::Real)
        isinteger(s) && zero(s) <= s || error("ns must be a non-negative integer")
        isinteger(f) && zero(f) <= f || error("nf must be a non-negative integer")
        isinteger(n) && zero(n) < n < s + f ||
            error("n must be a positive integer <= (ns + nf)")
        new(int(s), int(f), int(n))
    end
end

isupperbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true
islowerbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true
isbounded(d::Union(HyperGeometric, Type{HyperGeometric})) = true

min(d::HyperGeometric) = max(0,d.n-d.nf)
max(d::HyperGeometric) = min(d.n,d.ns)
support(d::HyperGeometric) = min(d):max(d)

insupport(d::HyperGeometric, x::Real) = isinteger(x) && zero(x) <= x <= d.n && (d.n - d.nf) <= x <= d.ns



mean(d::HyperGeometric) = d.n * d.ns / (d.ns + d.nf)

mode(d::HyperGeometric) = floor((d.n+1)*(d.ns+1)/(d.ns+d.nf+2))

function var(d::HyperGeometric)
    N = d.ns + d.nf
    d.n * (d.ns / N) * (d.nf / N) * ((N - d.n) / (N - 1.0))
end

function skewness(d::HyperGeometric)
    N = d.ns + d.nf
    (d.nf-d.ns)*((N-2d.n)/(N-2))*sqrt((N-1)/(d.n*d.ns*d.nf*(N-d.n)))
end

function kurtosis(d::HyperGeometric)
    N = d.ns + d.nf
    ((N-1)*N^2*(N*(N+1)-6*d.ns*d.nf-6*d.n*(N-d.n))+6d.n*d.ns*d.nf*(N-d.n)*(5N-6))/
    (d.n*d.ns*d.nf*(N-d.n)*(N-2)*(N-3))
end

function entropy(d::HyperGeometric)
    e = 0.0
    for x = support(d)
        p = pdf(d,x)
        e -= log(p)*p
    end
    e
end


function pdf(d::HyperGeometric, x::Real) 
    N = d.ns + d.nf
    p = d.ns / N
    pdf(Binomial(d.ns,p),x) * pdf(Binomial(d.nf,p),d.n-x) / pdf(Binomial(N,p),d.n)
end

function cdf(d::HyperGeometric, x::Real)
    if x < min(d)
        return 0.0
    elseif x >= max(d)
        return 1.0
    end
    p = 0.0
    for i = min(d):floor(x)
        p += pdf(d,i)
    end
    p
end

function quantile(d::HyperGeometric, p::Real)
    if p < 0 || p > 1 return NaN end
    if p == 0 return min(d) end
    if p == 1 return max(d) end
    cp = 0.0
    for x = support(d)
        cp += pdf(d,x)
        if cp >= p
            return x
        end
    end
end

# TODO: Implement:
#   V. Kachitvichyanukul & B. Schmeiser
#   "Computer generation of hypergeometric random variates"
#   Journal of Statistical Computation and Simulation, 22(2):127-145
#   doi:10.1080/00949658508810839
rand(d::HyperGeometric) = quantile(d,rand())
