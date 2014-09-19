push!(LOAD_PATH, "src")
sampler_tests = [
    "categorical",
    "binomial", 
    "poisson", 
    "exponential", 
    "gamma"]

tests = [
    "types",
    "fit", 
    "discrete",
    "univariate", 
    "truncate", 
    "multinomial",
    "dirichlet",
    "multivariate",
    "mvnormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "noncentralhypergeometric",
    "vonmisesfisher",
    "compoundvariate",
    "conjugates",
    "conjugates_normal",
    "conjugates_mvnormal",
    "wishart",
    "mixture",
    "gradlogpdf"]

println("Running tests:")

for t in sampler_tests
    test_fn = joinpath("samplers", "$t.jl")
    println(" * $test_fn")
    include(test_fn)
end

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    include(test_fn)
end
