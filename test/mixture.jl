using Distributions
using Base.Test

# Univariate Mixture

function test_univariate_mixture(g::UnivariateMixture, n::Int, ns::Int)
	X = zeros(n)
	for i = 1:10
		X[i] = rand(g) 
	end

	cs = components(g)
	pr = priorprobs(g)
	@assert length(cs) == length(pr)

	# mean
	mu = 0.0
	for k = 1:length(cs)
		mu += pr[k] * mean(cs[k])
	end
	@test_approx_eq mean(g) mu

	# ground-truth
	p0 = zeros(n)
	for i = 1:n
		p0_i = 0.0
		x_i = X[i]
		for k = 1:length(cs)
			p0_i += pr[k] * pdf(cs[k], x_i)
		end
		p0[i] = p0_i
	end
	lp0 = log(p0)

	for i = 1:n
		@test_approx_eq pdf(g, X[i]) p0[i]
		@test_approx_eq logpdf(g, X[i]) lp0[i]
	end

	p_e = pdf(g, X)
	lp_e = logpdf(g, X)
	@assert isa(p_e, Vector{Float64}) && length(p_e) == n
	@assert isa(lp_e, Vector{Float64}) && length(lp_e) == n

	@test_approx_eq p_e p0 
	@test_approx_eq lp_e lp0

	# sampling
	Xs = rand(g, ns)
	@test isa(Xs, Vector{Float64})
	@test_approx_eq_eps mean(Xs) mean(g) 0.01
end


println("    testing UnivariateMixture")

g_u = MixtureModel(
	Normal[Normal(0.0, 1.0), Normal(2.0, 1.0), Normal(-4.0, 1.5)], 
	[0.2, 0.5, 0.3])

@test isa(g_u, MixtureModel{Univariate, Continuous, Normal})
test_univariate_mixture(g_u, 1000, 10^6)


