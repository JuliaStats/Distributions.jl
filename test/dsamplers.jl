# Tests of Discrete samplers

using Distributions

function est_p(sampler, K::Int, n::Int)
	cnts = zeros(Int, K)
	for k = 1:n
		cnts[rand(sampler)] += 1 
	end
	cnts * inv(n)
end

function est_dev(p::Vector{Float64}, sampler, n::Int)
	max(abs(est_p(sampler, length(p), n) - p))
end

const m = 10
const K = 20
const n = 10^6

dev_naive = 0.
dev_dtable = 0.
dev_alias = 0.

for i = 1:m
	p = rand(K)
	p = p / sum(p)

	dev_naive  += est_dev(p, Categorical(p), n)
	dev_dtable += est_dev(p, Distributions.DiscreteDistributionTable(p), n)
	dev_alias  += est_dev(p, Distributions.AliasTable(p), n)

	print('.')
end
println()

dev_naive /= m
dev_dtable /= m
dev_alias /= m

function print_result(name, avg_dev)
	@printf("%-15s : avg.deviation = %.6e\n", name, avg_dev)
end

print_result("Naive", dev_naive)
print_result("DiscreteTable", dev_dtable)
print_result("AliasTable", dev_alias)
