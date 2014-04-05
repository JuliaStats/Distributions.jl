tests = [
    "cauchy",
    "geometric",
    "logistic", 
    "lognormal",
    "normal", 
    "uniform", 
    "weibull"]

println("Running rmath tests:")

for t in tests
    test_fn = joinpath("rmath", "$t.jl")
    println(" * $test_fn")
    include(test_fn)
end
