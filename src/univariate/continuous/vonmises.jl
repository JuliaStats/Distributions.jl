# von Mises distribution
#
#  Implemented based on Wikipedia
#

immutable VonMises <: ContinuousUnivariateDistribution
    μ::Float64      # mean
    κ::Float64      # concentration
    I0κ::Float64    # I0(κ), where I0 is the modified Bessel function of order 0

    function VonMises(μ::Real, κ::Real)
        κ > zero(κ) || error("kappa must be positive")
        @compat new(Float64(μ), Float64(κ), besseli(0, κ))
    end

    @compat VonMises(κ::Real) = VonMises(0.0, Float64(κ))
    VonMises() = VonMises(0.0, 1.0)
end

show(io::IO, d::VonMises) = show(io, d, (:μ, :κ))

@distr_support VonMises d.μ - π d.μ + π


#### Parameters

params(d::VonMises) = (d.μ, d.κ)


#### Statistics

mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
circvar(d::VonMises) = 1.0 - besseli(1, d.κ) / d.I0κ
entropy(d::VonMises) = log(twoπ * d.I0κ) - d.κ * (besseli(1, d.κ) / d.I0κ)

cf(d::VonMises, t::Real) = (besseli(abs(t), d.κ) / d.I0κ) * cis(t * d.μ)


#### Evaluation

pdf(d::VonMises, x::Float64) = exp(d.κ * cos(x - d.μ)) / (twoπ * d.I0κ)
logpdf(d::VonMises, x::Float64) = d.κ * cos(x - d.μ) - log(d.I0κ) - log2π

cdf(d::VonMises, x::Float64) = _vmcdf(d.κ, d.I0κ, x - d.μ, 1.0e-15)

function _vmcdf(κ::Float64, I0κ::Float64, x::Float64, tol::Float64)
    j = 1
    cj = besseli(j, κ) / j
    s = cj * sin(j * x)
    while abs(cj) > tol
        j += 1
        cj = besseli(j, κ) / j
        s += cj * sin(j * x)
    end
    return (x + 2.0 * s / I0κ) / twoπ + 0.5
end


#### Sampling

sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
