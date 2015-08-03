# TODO: this distribution may need clean-up

# Computes the pdf of a poisson-binomial random variable using
# fast fourier transform
#
#     Hong, Y. (2013).
#     On computing the distribution function for the Poisson binomial
#     distribution. Computational Statistics and Data Analysis, 59, 41–51.
#
function poissonbinomial_pdf_fft(p::AbstractArray)
    n = length(p)
    ω = 2. / (n + 1)

    x = Array(Complex{Float64}, n+1)
    @compat lmax = ceil(Int, n/2)
    x[1] = 1./(n+1)
    for l=1:lmax
        logz = 0.
        argz = 0.
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan2(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l+1] = dl * cos(argz) / (n+1) + dl * sin(argz) * im / (n+1)
        if n + 1 - l > l
            x[n+1-l+1] = conj(x[l+1])
        end
    end
    fft!(x)
    [max(0., real(xi)) for xi in x]
end

immutable PoissonBinomial <: DiscreteUnivariateDistribution

    p::Vector{Float64}
    pmf::Vector{Float64}
    function PoissonBinomial(p::AbstractArray)
        for i=1:length(p)
            if !(0.0 <= p[i] <= 1.0)
                error("Each element of p must be in [0, 1].")
            end
        end
        pb = poissonbinomial_pdf_fft(p)
        @assert isprobvec(pb)
        new(p, pb)
    end

end

@distr_support PoissonBinomial 0 length(d.p)

##### Parameters

ntrials(d::PoissonBinomial) = length(d.p)
succprob(d::PoissonBinomial) = d.p
failprob(d::PoissonBinomial) = 1. - d.p

params(d::PoissonBinomial) = (d.p, )

#### Properties

mean(d::PoissonBinomial) = sum(succprob(d))
var(d::PoissonBinomial) = sum(succprob(d) .* failprob(d))

function skewness(d::PoissonBinomial)
    v = 0.
    s = 0.
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1. - p[i])
        s += p[i] * (1. - p[i]) * (1. - 2. * p[i])
    end
    s / sqrt(v) / v
end

function kurtosis(d::PoissonBinomial)
    v = 0.
    s = 0.
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1. - p[i])
        s += p[i] * (1. - p[i]) * (1. - 6. * (1 - p[i] ) * p[i])
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
    prod(1. - p + p * exp(t))
end

function cf(d::PoissonBinomial, t::Real)
    p,  = params(d)
    prod(1. - p + p * cis(t))
end

pdf(d::PoissonBinomial, k::Int) = insupport(d, k) ? d.pmf[k+1] : 0.
logpdf(d::PoissonBinomial, k::Int) = insupport(d, k) ? log(d.pmf[k+1]) : -Inf
pdf(d::PoissonBinomial) = copy(d.pmf)

#### Sampling

sampler(d::PoissonBinomial) = PoissBinAliasSampler(d)
