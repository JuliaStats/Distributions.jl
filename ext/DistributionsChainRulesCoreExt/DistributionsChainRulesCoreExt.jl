module DistributionsChainRulesCoreExt

using Distributions
using Distributions: LinearAlgebra, SpecialFunctions, StatsFuns
import ChainRulesCore

include("eachvariate.jl")
include("utils.jl")

include("univariate/continuous/uniform.jl")
include("univariate/discrete/negativebinomial.jl")
include("univariate/discrete/poissonbinomial.jl")

include("multivariate/dirichlet.jl")

end # module
