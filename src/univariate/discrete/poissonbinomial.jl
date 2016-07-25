# TODO: this distribution may need clean-up
doc"""
    PoissonBinomial(p)

A *Poisson-binomial distribution* describes the number of successes in a sequence of independent trials, wherein each trial has a different success rate. It is parameterized by a vector `p` (of length ``K``), where ``K`` is the total number of trials and `p[i]` corresponds to the probability of success of the `i`th trial.

$P(X = k) = \sum\limits_{A\in F_k} \prod\limits_{i\in A} p[i] \prod\limits_{j\in A^c} (1-p[j]), \quad \text{ for } k = 0,1,2,\ldots,K,$

where $F_k$ is the set of all subsets of $k$ integers that can be selected from $\{1,2,3,...,K\}$.

```julia
PoissonBinomial(p)   # Poisson Binomial distribution with success rate vector p

params(d)            # Get the parameters, i.e. (p,)
succprob(d)          # Get the vector of success rates, i.e. p
failprob(d)          # Get the vector of failure rates, i.e. 1-p
```

External links:

* [Poisson-binomial distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_binomial_distribution)

"""
immutable PoissonBinomial{T<:Real} <: DiscreteUnivariateDistribution

    p::Vector{T}
    pmf::Vector{T}
    function PoissonBinomial(p::AbstractArray)
        for i=1:length(p)
            if !(0 <= p[i] <= 1)
                error("Each element of p must be in [0, 1].")
            end
        end
        pb = poissonbinomial_pdf_fft(p)
        @assert isprobvec(pb)
        new(p, pb)
    end

end

PoissonBinomial{T<:Real}(p::AbstractArray{T}) = PoissonBinomial{T}(p)

@distr_support PoissonBinomial 0 length(d.p)

#### Conversions

function PoissonBinomial{T <: Real, S <: Real}(::Type{PoissonBinomial{T}}, p::Vector{S})
    PoissonBinomial(Vector{T}(p))
end
function PoissonBinomial{T <: Real, S <: Real}(::Type{PoissonBinomial{T}}, d::PoissonBinomial{S})
    PoissonBinomial(Vector{T}(d.p))
end

#### Parameters

ntrials(d::PoissonBinomial) = length(d.p)
succprob(d::PoissonBinomial) = d.p
failprob(d::PoissonBinomial) = 1 - d.p

params(d::PoissonBinomial) = (d.p, )
@inline partype{T<:Real}(d::PoissonBinomial{T}) = T

#### Properties

mean(d::PoissonBinomial) = sum(succprob(d))
var(d::PoissonBinomial) = sum(succprob(d) .* failprob(d))

function skewness{T<:Real}(d::PoissonBinomial{T})
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1 - p[i])
        s += p[i] * (1 - p[i]) * (1 - 2 * p[i])
    end
    s / sqrt(v) / v
end

function kurtosis{T<:Real}(d::PoissonBinomial{T})
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1 - p[i])
        s += p[i] * (1 - p[i]) * (1 - 6 * (1 - p[i] ) * p[i])
    end
    s / v / v
end

entropy(d::PoissonBinomial) = entropy(Categorical(d.pmf))
median(d::PoissonBinomial) = median(Categorical(d.pmf)) - 1
mode(d::PoissonBinomial) = indmax(d.pmf) - 1
modes(d::PoissonBinomial) = [x  - 1 for x in modes(Categorical(d.pmf))]

#### Evaluation

quantile(d::PoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1

function mgf(d::PoissonBinomial, t::Real)
    p,  = params(d)
    prod(1 - p + p * exp(t))
end

function cf(d::PoissonBinomial, t::Real)
    p,  = params(d)
    prod(1 - p + p * cis(t))
end

pdf(d::PoissonBinomial, k::Int) = insupport(d, k) ? d.pmf[k+1] : 0
function logpdf{T<:Real}(d::PoissonBinomial{T}, k::Int)
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
pdf(d::PoissonBinomial) = copy(d.pmf)


# Computes the pdf of a poisson-binomial random variable using
# fast fourier transform
#
#     Hong, Y. (2013).
#     On computing the distribution function for the Poisson binomial
#     distribution. Computational Statistics and Data Analysis, 59, 41–51.
#
function poissonbinomial_pdf_fft(p::AbstractArray)
    n = length(p)
    ω = 2 / (n + 1)

    x = Array(Complex{Float64}, n+1)
    lmax = ceil(Int, n/2)
    x[1] = 1/(n + 1)
    for l=1:lmax
        logz = 0.
        argz = 0.
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan2(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l + 1] = dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
        if n + 1 - l > l
            x[n + 1 - l + 1] = conj(x[l + 1])
        end
    end
    fft!(x)
    [max(0, real(xi)) for xi in x]
end

#### Sampling

sampler(d::PoissonBinomial) = PoissBinAliasSampler(d)
