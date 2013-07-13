using Distributions
using Base.Test

probs = [0.2245, 0.1271, 0.3452, 0.3032]
drawtable = Distributions.DiscreteDistributionTable(probs)
rand(drawtable)

aliastable = Distributions.AliasTable(probs)
rand(aliastable)

for table = [drawtable,aliastable]

    N = 1_000_000
    results = Array(Int64, N)
    for i in 1:N
	results[i] = rand(table)
    end

    @test abs(sum(results .== 1) / N - probs[1]) < 0.1
    @test abs(sum(results .== 2) / N - probs[2]) < 0.1
    @test abs(sum(results .== 3) / N - probs[3]) < 0.1
    @test abs(sum(results .== 4) / N - probs[4]) < 0.1
end