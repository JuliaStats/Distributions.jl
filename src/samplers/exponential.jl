
randexp() = Base.Random.randmtzig_exprnd()

immutable ExponentialSampler <: Sampleable{Univariate,Continuous}
	scale::Float64
end

rand(s::ExponentialSampler) = s.scale * randexp()

