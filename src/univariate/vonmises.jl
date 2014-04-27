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

## Support
@continuous_distr_support VonMises -Inf Inf

## Properties
mean(d::VonMises) = d.μ
median(d::VonMises) = d.μ
mode(d::VonMises) = d.μ
var(d::VonMises) = 1.0 - besseliexpscaled(1, d.κ) / besseliexpscaled(0, d.κ)

function entropy(d::VonMises)
	I0κ = besseliexpscaled(0.0, d.κ)
	log(twoπ * I0κ) - d.κ * (besseliexpscaled(1, d.κ) / I0κ - 1.0)
end

## Functions
pdf(d::VonMises, x::Real) = exp(d.κ * (cos(x - d.μ) - 1.0)) / (twoπ * besseliexpscaled(0, d.κ))
logpdf(d::VonMises, x::Real) = d.κ * (cos(x - d.μ) - 1.0) - log2π - log(besseliexpscaled(0, d.κ))
cf(d::VonMises, t::Real) = besseliexpscaled(abs(t), d.k) / besseliexpscaled(0.0, d.κ) * exp(im * t * d.μ)
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
		sj = besseliexpscaled(j, κ) * sin(j * x) / j
		s += sj
		j += 1
		abs(sj) >= tol || break
	end
	x / twoπ + s / (π * besseliexpscaled(0, κ))
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

## Helper functions
# Bessel function as in Base/math.jl, but with exponential scaling
const cy = Array(Float64,2)
const ae = Array(Int32,2)
const openspecfun = "libopenspecfun"

type AmosException <: Exception
    info::Int32
end

# Computes modified bessel function of first kind, scaled by exp(-Re(z))
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

besseliexpscaled(nu::Float64, z::Complex128) = _besseliexpscaled(nu, z)
besseliexpscaled(nu::Real, z::Complex64) = complex64(besseliexpscaled(float64(nu), complex128(z)))
besseliexpscaled(nu::Real, z::Complex) = besseliexpscaled(float64(nu), complex128(z))
besseliexpscaled(nu::Real, x::Integer) = besseliexpscaled(nu, float64(x))
function besseliexpscaled(nu::Real, x::FloatingPoint)
    if x < 0 && !isinteger(nu)
        throw(DomainError())
    end
    oftype(x, real(besseliexpscaled(float64(nu), complex128(x))))
end
