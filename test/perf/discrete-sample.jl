using Distributions

n = 1_000_000
p = fill(1/n,n)
pu = fill(uint(1),n)

println("Set-up condensed table")
d = Distributions.DiscreteDistributionTable(p)
gc()
@time d = Distributions.DiscreteDistributionTable(p)

println("Set-up Huffman tree")
h = Distributions.huffman(1:length(pu),pu)
gc()
@time h = Distributions.huffman(1:length(pu),pu)

println("Set-up alias table")
a = Distributions.AliasTable(p)
gc()
@time a = Distributions.AliasTable(p)

function runsample(d,N)
    x = Array(Int,N)
    for i = 1:N
        x[i] = rand(d) 
    end
    x
end


println("Sample condensed table")
x = runsample(d,10)
gc()
@time x = runsample(d,1_000_000)

println("Sample Huffman tree")
x = runsample(h,10)
gc()
@time x = runsample(h,1_000_000)

println("Sample alias table")
x = runsample(a,10)
gc()
@time x = runsample(a,1_000_000)
