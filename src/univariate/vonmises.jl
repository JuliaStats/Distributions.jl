# von Mises distribution

immutable VonMises <: ContinuousUnivariateDistribution
    μ::Float64  # mean
    κ::Float64  # concentration

    function VonMises(μ::Real, κ::Real)
        κ > zero(κ) || error("kappa must be positive")
        new(float64(μ), float64(κ))
    end

    VonMises(κ::Real) = VonMises(0.0, float64(κ))
    VonMises() = VonMises(0.0, 1.0)
end

## Properties
circmean(d::VonMises) = d.μ
circmedian(d::VonMises) = d.μ
circmode(d::VonMises) = d.μ
circvar(d::VonMises) = 1.0 - besselix(1, d.κ) / besselix(0, d.κ)

function entropy(d::VonMises)
	I0κ = besselix(0.0, d.κ)
	log(twoπ * I0κ) - d.κ * (besselix(1, d.κ) / I0κ - 1.0)
end

## Functions
pdf(d::VonMises, x::Real) = exp(d.κ * (cos(x - d.μ) - 1.0)) / (twoπ * besselix(0, d.κ))
logpdf(d::VonMises, x::Real) = d.κ * (cos(x - d.μ) - 1.0) - log2π - log(besselix(0, d.κ))
cf(d::VonMises, t::Real) = besselix(abs(t), d.k) / besselix(0.0, d.κ) * exp(im * t * d.μ)
cdf(d::VonMises, x::Real) = cdf(d, x, d.μ - π)

function cdf(d::VonMises, x::Real, from::Real)
	const tol = 1.0e-20
	x = mod(x - from, twoπ)
	μ = mod(d.μ - from, twoπ)
	if μ == 0.0
		return vmcdfseries(d.κ, x, tol)
	elseif x <= μ
		upper = mod(x - μ, twoπ)
		if upper == 0.0
			upper = twoπ
		end
		return vmcdfseries(d.κ, upper, tol) - vmcdfseries(d.κ, mod(-μ, twoπ), tol)
	else
		return vmcdfseries(d.κ, x - μ, tol) - vmcdfseries(d.κ, μ, tol)
	end
end

function vmcdfseries(κ::Real, x::Real, tol::Real)
	j, s = 1, 0.0
	while true
		sj = besselix(j, κ) * sin(j * x) / j
		s += sj
		j += 1
		abs(sj) >= tol || break
	end
	x / twoπ + s / (π * besselix(0, κ))
end

## Sampling
sampler(d::VonMises) = VonMisesSampler(d.μ, d.κ)
