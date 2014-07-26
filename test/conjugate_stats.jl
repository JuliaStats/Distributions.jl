using Distributions
using Base.Test

const n_samples = 5_000_001

for d in [
	NormalGamma(-1.0, 1.0, 3.0, 1.0),
	NormalGamma( 0.0, 1.0, 3.0, 1.0),
    NormalGamma( 0.0, 5.0, 3.0, 1.0),
    NormalGamma( 0.0, 5.0, 4.0, 1.0),
    NormalGamma( 0.0,10.0, 4.0, 2.0),
    NormalInverseGamma(-2.0, 1.0, 3.0, 1.0),
    NormalInverseGamma( 0.0, 1.0, 3.0, 1.0),
    NormalInverseGamma( 0.0, 0.2, 3.0, 1.0),
    NormalInverseGamma( 0.0, 0.2, 4.0, 1.0),
    NormalInverseGamma( 0.0, 0.1, 4.0, 2.0)]

	println(d)
	dmean = mean(d)
	dvar = var(d)

	xmu   = Array(Float64, n_samples)
	xv2 = Array(Float64, n_samples)
	for i = 1 : n_samples
		xmu[i], xv2[i] = rand(d)
	end
	mumean = mean(xmu)
	v2mean = mean(xv2)
	muvar  = var(xmu)
	v2var  = var(xv2)

	@test_approx_eq_eps mumean dmean[1]  0.01
	@test_approx_eq_eps v2mean dmean[2]  0.01
	@test_approx_eq_eps muvar  dvar[1]   0.02
	@test_approx_eq_eps v2var  dvar[2]   0.02
end
