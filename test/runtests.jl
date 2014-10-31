tests = [
    "types",
    "samplers",
    "fit", 
    "univariates", 
    "truncate", 
    "multinomial",
    "dirichlet",
    "hypergeometric",
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

print_with_color(:blue, "Running tests:\n")

for t in tests
    test_fn = "$t.jl"
    print_with_color(:green, "* $test_fn\n")
    include(test_fn)
end
