@testset "Generalized inverse Gaussian" begin
	d = GeneralizedInverseGaussian(3, 8, -1/2)
	samples = rand(d, 1000_000)

	@test abs(mean(d) - mean(samples)) < 0.01
	@test abs(var(d) - var(samples)) < 0.01

	test_distr(d, 10^6, testquan=false) # TODO: implement quantiles
end