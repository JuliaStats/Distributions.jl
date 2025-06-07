import SpecialFunctions: besselk

@testset "Generalized inverse Gaussian" begin
	GIG(μ, λ, θ) = GeneralizedInverseGaussian(Val(:Wolfram), μ, λ, θ)
	# Derivative d/dp log(besselk(p, x))
	dlog_besselk_dp(p::Real, x::Real, h::Real=1e-5) = (log(besselk(p + h, x)) - log(besselk(p - h, x))) / (2h)

	distributions = [
		# basic inverse Gaussian
		(
			d=GIG(3, 8, -1/2),
			tmean=3,
			tvar=3.375,
			tskew=1.83711730708738357364796305603,
			tkurt=5.625
		), (
			d=GIG(3, 8, 1.0),
			tmean=4.80482214427053451026053858645,
			tvar=7.53538381114490814886717269870,
			tskew=1.46944720741022099816138306944,
			tkurt=3.41236660328772156010103826417
		), (
			d=GIG(1, 1, 1.0),
			tmean=2.69948393559377234389267739977,
			tvar=4.51072222384624734344698799307,
			tskew=1.80431858973895795576206086081,
			tkurt=4.99738802301388098591337070387
		), (
			d=GIG(1, 1, -1.0),
			tmean=0.699483935593772343892677399771,
			tvar=0.510722223846247343446987993073,
			tskew=3.52208925791368986307687496050,
			tkurt=21.7757675389062712238907921144
		)
	]
	NSAMPLES = 10^6
	for (d, mean_true, var_true, skew_true, kurt_true) in distributions
		println("\ttesting $d")

		@test collect(params(d)) ≈ [d.a, d.b, d.p]

		@test mean(d)     ≈ mean_true
		@test var(d)      ≈ var_true
		@test skewness(d) ≈ skew_true
		@test kurtosis(d) ≈ kurt_true

		samples = rand(d, NSAMPLES)

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

		test_samples(d, NSAMPLES)
	end
end
