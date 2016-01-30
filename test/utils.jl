using Distributions
using Base.Test

# RealInterval
for (l,u,lc,uc) in [(:closed,:closed,<=,<=),
                    (:open,:closed,<,<=),
                    (:closed,:open,<=,<),
                    (:open,:open,<,<)]

  r = RealInterval(1.5, 4.0, l, u)
  @test minimum(r) == 1.5
  @test maximum(r) == 4.0
  @test Distributions.lowerboundary(r) == l
  @test Distributions.upperboundary(r) == u
  @test Distributions.lowercomparator(r) == lc
  @test Distributions.uppercomparator(r) == uc

end

@test RealInterval(0.0,1.0) == RealInterval(0.0,1.0,:closed,:closed)

