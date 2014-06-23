# delegation of samplers

for fname in ["categorical.jl",
              "binomial.jl",
              "poisson.jl", 
              "exponential.jl",
              "gamma.jl", 
              "multinomial.jl",
              "vonmises.jl"]
              
    include(joinpath("samplers", fname))
end
