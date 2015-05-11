immutable MaxwellBoltzmann <: ContinuousUnivariateDistribution
    a::Float64 # Scale factor
    chisqd::Chisq

    function MaxwellBoltzmann(a::Real)
        a > zero(a) || error("MaxwellBoltzmann: a must be positive")
        @compat new(Float64(a), Chisq(3.0))
    end

    function MaxwellBoltzmann(m::Real, T::Real) # mass and temperature of particles
        m > zero(m) && T > zero(T) || error("MaxwellBoltzmann: m and T must be positive")
        @compat new(Float64(sqrt(1.3806488e-23*T/m)), Chisq(3.0))
    end
    MaxwellBoltzmann() = new(1.0,Chisq(3.0))
end

@distr_support MaxwellBoltzmann 0.0 Inf

#### Parameters

scale(d::MaxwellBoltzmann) = d.a
params(d::MaxwellBoltzmann) = (d.a,)

#### Statistics

mean(d::MaxwellBoltzmann) = 1.5957691216057308 * d.a # 2*sqrt(2/π)
median(d::MaxwellBoltzmann) = 1.5381722544550522 * d.a # quantile(d, 0.5)
mode(d::MaxwellBoltzmann) = sqrt2 * d.a # sqrt(2) = 1.4142135623730951

var(d::MaxwellBoltzmann) = 0.4535209105296745 * d.a^2 # (3.0π - 8.0)/π
std(d::MaxwellBoltzmann) = 0.6734396116428515 * d.a

skewness(d::MaxwellBoltzmann) = 0.4856928280495921 # 2.0 * sqrt(2.0) * (16.0 - 5.0π)/(3π - 8)^(3/2)
kurtosis(d::MaxwellBoltzmann) = 0.10816384281628826 # 4.0 * (-96.0 + 40.0π - 3π^2)/(3.0π - 8.0)^2

entropy(d::MaxwellBoltzmann) = 0.9961541981062054 + log(d.a) # log(sqrt(2π)) + γ - 0.5

#### Evaluation

pdf(d::MaxwellBoltzmann, x::Float64) = exp(logpdf(d, x))

function logpdf(d::MaxwellBoltzmann, x::Float64)
    x > 0.0 ? logsqrt2onπ + 2.0 * log(x) - 0.5 * x^2 / d.a^2 - 3.0 * log(d.a) : -Inf
end

cdf(d::MaxwellBoltzmann, x::Float64) = cdf(d.chisqd, (x/d.a)^2)
logcdf(d::MaxwellBoltzmann, x::Float64) = logcdf(d.chisqd, (x/d.a)^2)

gradlogpdf(d::MaxwellBoltzmann, x::Float64) = x >= 0.0 ? 2.0 / x - x/d.a^2 : 0.0

quantile(d::MaxwellBoltzmann, p::Float64) = quantile_newton(d, p)

#### Sampling

rand(d::MaxwellBoltzmann) = d.a * sqrt(rand(d.chisqd))

#### Fitting

function fit_mle{T <: Real}(::Type{MaxwellBoltzmann}, x::Array{T})
    a = sqrt(1.0/(3.0*length(x)) * sumabs2(x))
    MaxwellBoltzmann(a)
end

function fit_mle{T <: Real}(::Type{MaxwellBoltzmann}, x::Array{T}, w::Array{T})
    a = sqrt(1.0/(3.0*sum(w)) * sum(w.*x.^2))
    MaxwellBoltzmann(a)
end
