using Test
using Distributions, Random
import Distributions.fold_value
import Distributions.unfold_value


# Constructor works
@test folded(Normal(1.0,2.0),1.0) isa Folded

# Testing if the fold_value function and unfold_value functions fold the 
# given value properly
@testset "Folding and unfolding value in the domain" begin
	@test fold_value(-10, folded(Normal(3,4),0)) == 10
	@test fold_value( 5 , folded(Normal(3,4),0)) == 5
	@test fold_value( 4, folded(Normal(2,3),10, keep_right=false)) == 4
	@test fold_value( 16, folded(Normal(2,3),10, keep_right=false)) == 4
	@test unfold_value(10, folded(Normal(3,4),0)) == -10
	@test unfold_value( 5 , folded(Normal(3,4),0)) == -5
	@test unfold_value( 16, folded(Normal(2,3),10, keep_right=false)) == 16
	@test unfold_value( 4, folded(Normal(2,3),10, keep_right=false)) == 16
end 

# Testing if the insupport function works
@testset "Folded insupport, minimum and maximum functions" begin
	@test insupport(folded(Normal(2,1),0),-10) == false
	# Check that the crease is always the minimum of a folded distrbution
	@test all(((x->Distributions.minimum(folded(Normal(2,1),x))).(-10:0.1:10)) .≈ collect(-10:0.1:10))
	# Check that the crease is always the maximum of a distrbution folded to the right
	@test all(((x->Distributions.maximum(folded(Normal(2,1),x,keep_right=false))).(-10:0.1:10)) .≈ collect(-10:0.1:10))
end

@testset "Folded Rand functions" begin
	# Check that a folded distribution with a crease at 0 always generates positive numbers
	@test all(rand(folded(Normal(1.0,1.0),0.0),100) .> 0.0)
end

@testset "Folded pdf, logpdf, cdf and logcdf functions" begin
	@test pdf(truncated(Normal(3,4),-1,1),-0.1) + pdf(truncated(Normal(3,4),-1,1),0.1) ≈ pdf(folded(truncated(Normal(3,4),-1,1),0.0),0.1)
	@test pdf(folded(truncated(Normal(3,4),-1,1),0.0),-0.1) == 0
end

# Test taken from the truncated tests
@testset "Folded Distribution should be numerically stable at low probability regions" begin
    original = Normal(-5.0, 0.2)
    trunc = folded(original, 0.0)
    for x in LinRange(0.0, 5.0, 100)
        @test isfinite(logpdf(original, x))
        @test isfinite(logpdf(trunc, x))
        @test isfinite(pdf(trunc, x))
    end
end