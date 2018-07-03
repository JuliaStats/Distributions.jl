using Distributions
using Test

a = qqbuild(collect(1:10), collect(1:10))
@test a.qx == collect(1.0:10)
@test a.qy == collect(1.0:10)

a = qqbuild(collect(1:10), Uniform(1,10))
@test a.qx == collect(2.0:9)
@test a.qy == collect(2.0:9)

a = qqbuild(Uniform(1,10), collect(1:10))
@test a.qx == collect(2.0:9)
@test a.qy == collect(2.0:9)
