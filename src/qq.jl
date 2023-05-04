struct QQPair{U<:AbstractVector, V<:AbstractVector}
    qx::U
    qy::V
end

function qqbuild(x::AbstractVector, y::AbstractVector)
    n = min(length(x), length(y))
    grid = [0.0:(1 / (n - 1)):1.0;]
    qx = quantile(x, grid)
    qy = quantile(y, grid)
    return QQPair(qx, qy)
end


function qqbuild(x::AbstractVector, d::UnivariateDistribution)
    ## Follows Matlab's convention (https://ch.mathworks.com/help/stats/qqplot.html)
    n = length(x)
    grid = ((1:n) .- 0.5) ./ n
    qd = quantile.(Ref(d), grid)
    return QQPair(qx, qd)
end

function qqbuild(d::UnivariateDistribution, x::AbstractVector)
    ## Follows Matlab's convention (https://ch.mathworks.com/help/stats/qqplot.html)
    n = length(x)
    grid = ((1:n) .- 0.5) ./ n
    qd = quantile.(Ref(d), grid)
    qx = quantile(x, grid)
    return QQPair(qd, qx)
end
