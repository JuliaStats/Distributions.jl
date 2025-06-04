import SpecialFunctions: besselk

raw"""
    GeneralizedInverseGaussian(a, b, p)

The *generalized inverse Gaussian distribution* with parameters `a>0`, `b>0` and real `p` has probability density function:

```math
f(x; a, b, p) =
\frac{(a/b)^(p/2)}{2 K_p(\sqrt{ab})}
x^{p-1} e^{-(ax + b/x)/2}, \quad x > 0
```

External links:

* [Generalized Inverse Gaussian distribution on Wikipedia](https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution).
"""
struct GeneralizedInverseGaussian{T1<:Real, T2<:Real, T3<:Real} <: ContinuousUnivariateDistribution
	a::T1
	b::T2
	p::T3
	function GeneralizedInverseGaussian(a::T1, b::T2, p::T3) where {T1<:Real, T2<:Real, T3<:Real}
		@assert a >= 0
		@assert b >= 0
		new{T1, T2, T3}(a, b, p)
	end
end

"""
    GeneralizedInverseGaussian(; μ::Real, λ::Real, θ::Real=-1/2)

Wolfram Language parameterization, equivalent to `InverseGamma(μ, λ)`
"""
GeneralizedInverseGaussian(; μ::Real, λ::Real, θ::Real=-1/2) =
	GeneralizedInverseGaussian(λ / μ^2, λ, θ)

params(d::GeneralizedInverseGaussian) = (d.a, d.b, d.p)
minimum(d::GeneralizedInverseGaussian) = 0.0
miximum(d::GeneralizedInverseGaussian) = Inf
insupport(d::GeneralizedInverseGaussian, x::Real) = x >= 0

mode(d::GeneralizedInverseGaussian) = (
	(d.p - 1) + sqrt((d.p - 1)^2 + d.a * d.b)
) / d.a

mean(d::GeneralizedInverseGaussian) =
	sqrt(d.b/d.a) * besselk(d.p+1, sqrt(d.a*d.b)) / besselk(d.p, sqrt(d.a*d.b))

var(d::GeneralizedInverseGaussian) = begin
	tmp1 = sqrt(d.a * d.b)
	tmp2 = besselk(d.p, tmp1)
	d.b/d.a * (
		besselk(d.p+2, tmp1) / tmp2 - (besselk(d.p+1, tmp1) / tmp2)^2
	)
end

logpdf(d::GeneralizedInverseGaussian, x::Real) = (
	d.p / 2 * log(d.a / d.b) - log(2 * besselk(d.p, sqrt(d.a * d.b)))
	+ (d.p - 1) * log(x) - (d.a * x + d.b / x) / 2
)

cdf(d::GeneralizedInverseGaussian, x::Real) = quadgk(
	z -> pdf(d, z), 0, x, maxevals=1000
)[1]

mgf(d::GeneralizedInverseGaussian, t::Real) =
	(d.a / (d.a - 2t))^(d.p/2) * (
		besselk(d.p+1, sqrt(d.a * d.b)) / besselk(d.p, sqrt(d.a * d.b))
	)

cf(d::GeneralizedInverseGaussian, t::Number) =
	(d.a / (d.a - 2im * t))^(d.p/2) * (
		besselk(d.p, sqrt(d.b * (d.a - 2t))) / besselk(d.p, sqrt(d.a * d.b))
	)

rand(rng::Random.AbstractRNG, d::GeneralizedInverseGaussian) = begin
	# Paper says ω = sqrt(b/a), but Wolfram disagrees
	ω = sqrt(d.a * d.b)
	sqrt(d.b / d.a) * rand(rng, _GIG(d.p, ω))
end

# ===== Private two-parameter version =====
"""
    _GIG(λ, ω)

Two-parameter generalized inverse Gaussian distribution, only used for sampling.

If `X ~ GeneralizedInverseGaussian(a, b, p)`, then `Y = sqrt(a/b) * X` follows `_GIG(p, 2sqrt(b * a))`.
NOTE: the paper says (Section 1) that the second parameter of `_GIG` should be `ω = 2sqrt(b/a)`, but computations in Wolfram Mathematica show otherwise.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
struct _GIG{T1<:Real, T2<:Real} <: ContinuousUnivariateDistribution
	λ::T1
	ω::T2
	function _GIG(λ::T1, ω::T2) where {T1<:Real, T2<:Real}
		@assert ω >= 0
		new{T1, T2}(λ, ω)
	end
end

logpdf(d::_GIG, x::Real) =
	if x > 0
		-log(2 * besselk(-d.λ, d.ω)) + (d.λ - 1) * log(x) - d.ω/2 * (x + 1/x)
	else
		-Inf
	end

cdf(d::_GIG, x::Real) = quadgk(
	z -> pdf(d, z), 0, x, maxevals=1000
)[1]

mean(d::_GIG) = besselk(1 + d.λ, d.ω) / besselk(d.λ, d.ω)
var(d::_GIG) = begin
	tmp = besselk(d.λ, d.ω)
	(
		tmp * besselk(2 + d.λ, d.ω) - besselk(1 + d.λ, d.ω)^2
	) / tmp^2
end

"""
    rand(rng::Random.AbstractRNG, d::_GIG)

Sampling from the _2-parameter_ generalized inverse Gaussian distribution based on [1], end of Section 6.

### References

1. Devroye, Luc. 2014. “Random Variate Generation for the Generalized Inverse Gaussian Distribution.” Statistics and Computing 24 (2): 239–46. https://doi.org/10.1007/s11222-012-9367-z.
"""
function rand(rng::AbstractRNG, d::_GIG)
	λ, ω = d.λ, d.ω
	(λ < 0) && return 1 / rand(rng, _GIG(-λ, ω))
	
	α = sqrt(ω^2 + λ^2) - λ
	ψ(x) = -α * (cosh(x) - 1) - λ * (exp(x) - x - 1)
	ψprime(x) = -α * sinh(x) - λ * (exp(x) - 1)

	tmp = -ψ(1)
	t = if 0.5 <= tmp <= 2
		1.0
	elseif tmp > 2
		sqrt(2 / (α + λ))
	else
		log(4 / (α + 2λ))
	end

	tmp = -ψ(-1)
	s = if 0.5 <= tmp <= 2
		1.0
	elseif tmp > 2
		sqrt(4 / (α * cosh(1) + λ))
	else
		min(1/λ, log(1 + 1/α + sqrt(1 / α^2 + 2/α)))
	end

	eta, zeta, theta, xi = -ψ(t), -ψprime(t), -ψ(-s), ψprime(-s)
	p, r = 1/xi, 1/zeta

	t_ = t - r * eta
	s_ = s - p * theta
	q = t_ + s_

	chi(x) = if -s_ <= x <= t_
		1.0
	elseif x < -s_
		exp(-theta + xi * (x + s))
	else # x > t_
		exp(-eta - zeta * (x - t))
	end

	# Generation
	UVW = rand(rng, 3) # allocates 3 x Float64
	U, V, W = UVW
	X = if U < q / (p + q + r)
		-s_ + q * V
	elseif U < (q + r) / (p + q + r)
		t_ - r * log(V)
	else
		-s_ + p * log(V)
	end
	while W * chi(X) > exp(ψ(X))
		Random.rand!(UVW)
		U, V, W = UVW
		X = if U < q / (p + q + r)
			-s_ + q * V
		elseif U < (q + r) / (p + q + r)
			t_ - r * log(V)
		else
			-s_ + p * log(V)
		end
	end

	tmp = λ/ω
	(tmp + sqrt(1 + tmp^2)) * exp(X)
end