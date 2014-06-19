
for fname in ["dirichlet.jl",
              "multinomial.jl",
              "mvnormal.jl", 
              "mvnormalcanon.jl",
              "mvtdist.jl",
              "vonmisesfisher.jl"]
    include(joinpath("multivariate", fname))
end
