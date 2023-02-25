module DistributionsChainRulesCoreExt

using Distributions
using Distributions: SpecialFunctions, StatsFuns
import ChainRulesCore

include("eachvariate.jl")
include("multivariate/dirichlet.jl")
include("univariate/continuous/uniform.jl")
include("univariate/discrete/negativebinomial.jl")
include("univariate/discrete/poissonbinomial.jl")

end # module
