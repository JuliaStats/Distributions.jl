# delegation of samplers

for fname in ["categorical.jl",
              "binomial.jl",
              "poisson.jl", 
              "exponential.jl",
              "gamma.jl"]
    include(joinpath("samplers", fname))
end
