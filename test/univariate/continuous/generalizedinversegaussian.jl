import SpecialFunctions: besselk

@testset "Generalized inverse Gaussian" begin
	# Derivative d/dp log(besselk(p, x))
	dlog_besselk_dp(p::Real, x::Real, h::Real=1e-5) = (log(besselk(p + h, x)) - log(besselk(p - h, x))) / (2h)

	N = 10^6
	for d in [
		GeneralizedInverseGaussian(3, 8, -1/2), # basic inverse Gaussian
		GeneralizedInverseGaussian(3, 8, 1.0),
		GeneralizedInverseGaussian(1, 1, 1.0),
		GeneralizedInverseGaussian(1, 1, -1.0),
	]
		println("\ttesting $d")
		samples = rand(d, N)

		@test abs(mean(d) - mean(samples)) < 0.01
		@test abs(std(d) - std(samples)) < 0.01

		a, b, p = params(d)
		t = sqrt(a * b)
		# E[log p(x; a,b,p)]
		expected_loglik_true = (
			p/2 * log(a/b) - log(2besselk(p, t))
			+ (p-1) * (0.5 * log(b/a) + dlog_besselk_dp(p, t))
			- (t * besselk(p+1, t)/besselk(p, t) - p)
		)
		expected_loglik_sample = mean(x->logpdf(d, x), samples)
		@test abs(expected_loglik_true - expected_loglik_sample) < 0.01

		test_samples(d, N)
	end
end
