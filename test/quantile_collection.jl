using Test
using StatsBase, Statistics, Distributions

# tests that quantile(collection) still works by way of iqr(collection)

@testset "Quantile of Collection" begin
    numbers, weights = 0:9, repeat(0.1, 10)
    avg_num = mean(numbers, weights)
    
    stddev = std(numbers, weights, mean=avg_num)
    num_dist = Normal(avg_num, stdev)
    
    range = iqr(num_dist)
end
