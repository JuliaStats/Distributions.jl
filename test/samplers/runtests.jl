
tests = ["categorical", 
		 "binomial", 
		 "poisson", 
		 "exponential", 
		 "gamma"]

println("Testing samplers:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end

