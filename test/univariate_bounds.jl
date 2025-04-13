using Distributions
using Test

# to make sure that subtypes provides the required behavior without having to add
# a dependency to InteractiveUtils
function _subtypes(m::Module, x::Type, sts=Base.IdSet{Any}(), visited=Base.IdSet{Module}())
    push!(visited, m)
    xt = Base.unwrap_unionall(x)
    if !isa(xt, DataType)
        return sts
    end
    xt = xt::DataType
    for s in names(m, all = true)
        if isdefined(m, s) && !Base.isdeprecated(m, s)
            t = getfield(m, s)
            if isa(t, DataType)
                t = t::DataType
                if t.name.name === s && supertype(t).name == xt.name
                    ti = typeintersect(t, x)
                    ti != Base.Bottom && push!(sts, ti)
                end
            elseif isa(t, UnionAll)
                t = t::UnionAll
                tt = Base.unwrap_unionall(t)
                isa(tt, DataType) || continue
                tt = tt::DataType
                if tt.name.name === s && supertype(tt).name == xt.name
                    ti = typeintersect(t, x)
                    ti != Base.Bottom && push!(sts, ti)
                end
            elseif isa(t, Module)
                t = t::Module
                in(t, visited) || _subtypes(t, x, sts, visited)
            end
        end
    end
    return sts
end

function _subtypes_in(mods::Array, x::Type)
    if !isabstracttype(x)
        # Fast path
        return Type[]
    end
    sts = Base.IdSet{Any}()
    visited = Base.IdSet{Module}()
    for m in mods
        _subtypes(m, x, sts, visited)
    end
    return sort!(collect(sts), by=string)
end

get_subtypes(m::Module, x::Type) = _subtypes_in([m], x)

get_subtypes(x::Type) = _subtypes_in(Base.loaded_modules_array(), x)


dists = get_subtypes(UnivariateDistribution)
filter!(x -> hasmethod(x, ()), dists)
filter!(x -> isbounded(x()), dists)

@testset "bound checking $dist" for dist in dists
    d = dist()
    lb,ub = float.(extrema(support(d)))
    lb = prevfloat(lb)
    ub = nextfloat(ub)
    @test iszero(cdf(d, lb))
    @test isone(cdf(d, ub))
    @test iszero(cdf(d, -Inf))
    @test isone(cdf(d, Inf))
    @test isnan(cdf(d, NaN))

    lb_lcdf = logcdf(d,lb)
    @test isinf(lb_lcdf) & (lb_lcdf < 0)
    @test iszero(logcdf(d, ub))
    @test logcdf(d, -Inf) == -Inf
    @test iszero(logcdf(d, Inf))
    @test isnan(logcdf(d, NaN))

    @test isone(ccdf(d, lb))
    @test iszero(ccdf(d, ub))
    @test isone(ccdf(d, -Inf))
    @test iszero(ccdf(d, Inf))
    @test isnan(ccdf(d, NaN))

    ub_lccdf = logccdf(d,ub)
    @test isinf(ub_lccdf) & (ub_lccdf < 0)
    @test iszero(logccdf(d, lb))
    @test iszero(logccdf(d, -Inf))
    @test logccdf(d, Inf) == -Inf
    @test isnan(logccdf(d, NaN))

    @test iszero(pdf(d, lb))
    @test iszero(pdf(d, ub))

    lb_lpdf = logpdf(d, lb)
    @test isinf(lb_lpdf) && lb_lpdf < 0
    ub_lpdf = logpdf(d, ub)
    @test isinf(ub_lpdf) && ub_lpdf < 0
    @test logpdf(d, -Inf) == -Inf
    @test logpdf(d, Inf) == -Inf
end
