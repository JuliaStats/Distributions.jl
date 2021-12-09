# Tests on Univariate LogLogistic distribusions 

using Distributions 
using Random, Test

#### 

isnan_type(::Type{T}, v) where {T} = isnan(v) && v isa T

@testset "LogLogistic" begin
	@test isa(convert(LogLogistic{Float64}, Float16(1), Float16(3)), LogLogistic{Float64}) 
	@test mean(LogLogistic(sin(1), pi)) ≈ 1.0 rtol = 1e-12  
	@test mode(LogLogistic(sqrt(3), 2)) ≈ 1.0 rtol = 1e-12
	@test var(LogLogistic(1, pi)) ≈ 0.7872174131518408 rtol = 1e-12  
	@test pdf(LogLogistic(1,3),-1) == 0.0
	@test logpdf(LogLogistic(1,3),-1) == -Inf
	@test cdf(LogLogistic(1,3),0) == 0.0 
	@test ccdf(LogLogistic(1,3),0) == 1.0
	@test logccdf(LogLogistic(1,3),0) == 0.0 
end 
