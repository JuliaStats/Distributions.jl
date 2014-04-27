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
# normal approximation for large concentrations
rand(d::VonMises) = mod(d.μ + 
	(d.κ > 700.0 ? sqrt(1.0 / d.κ) * randn() : vmrand(d.κ)), twoπ)

# from Best & Fisher (1979): Efficient Simulation of the von Mises Distribution
function vmrand(κ::Float64)
	const tau = 1.0 + sqrt(1.0 + 4 * κ ^ 2)
	const rho = (tau - sqrt(2.0 * tau)) / (2.0 * κ)
	const r = (1.0 + rho ^ 2) / (2.0 * rho)

	f = 0.0
	while true
		t, u = 0.0, 0.0
		while true
		    const v, w = rand() - 0.5, rand() - 0.5
		    const d, e = v ^ 2, w ^ 2
		    if d + e <= 0.25
		    	t = d / e
		    	u = 4 * (d + e)
		    	break
		    end
		end
		const z = (1.0 - t) / (1.0 + t)
		f = (1.0 + r * z) / (r + z)
		const c = κ * (r - f)
		if c * (2.0 - c) > u || log(c / u) + 1 >= c
			break
		end
	end
	rand() > 0.5 ? acos(f) : -acos(f)
end

# TODO: remove as soon as implemented in Base/math.jl
## Helper functions
# Bessel function as in Base/math.jl, but with exponential scaling
import Base.Math: cy, ae, openspecfun, AmosException

# Computes modified bessel function of first kind, scaled by exp(-|Re(z)|)
function _besseliexpscaled(nu::Float64, z::Complex128)
    ccall((:zbesi_,openspecfun), Void,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          &real(z), &imag(z), &nu, &2, &1,
          pointer(cy,1), pointer(cy,2),
          pointer(ae,1), pointer(ae,2))
    if ae[2] == 0 || ae[2] == 3 
        return complex(cy[1],cy[2]) 
    else
        throw(AmosException(ae[2]))
    end
end

besselix(nu::Float64, z::Complex128) = _besseliexpscaled(nu, z)
besselix(nu::Real, z::Complex64) = complex64(besselix(float64(nu), complex128(z)))
besselix(nu::Real, z::Complex) = besselix(float64(nu), complex128(z))
besselix(nu::Real, x::Integer) = besselix(nu, float64(x))
function besselix(nu::Real, x::FloatingPoint)
    if x < 0 && !isinteger(nu)
        throw(DomainError())
    end
    oftype(x, real(besselix(float64(nu), complex128(x))))
end
