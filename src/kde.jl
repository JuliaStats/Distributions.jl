# Silverman's rule of thumb for KDE bandwidth selection
function bandwidth(data::Vector, alpha::Float64 = 0.9)
    # Determine length of data
    ndata = length(data)

    # Calculate width using variance and IQR
    var_width = std(data)
    q25, q75 = quantile(data, [0.25, 0.75])
    quantile_width = (q75 - q25) / 1.34

    # Deal with edge cases with 0 IQR or variance
    width = min(var_width, quantile_width)
    if width == 0.0
        if var_width == 0.0
            width = 1.0
        else
            width = var_width
        end
    end

    # Set bandwidth using Silverman's rule of thumb
    return alpha * width * ndata^(-0.2)
end

# Store both grid and density for KDE over the real line
immutable UnivariateKDE
    x::Vector{Float64}
    density::Vector{Float64}
end

# Store both grid and density for KDE over R2
immutable BivariateKDE
    x::Vector{Float64}
    y::Vector{Float64}
    density::Matrix{Float64}
end

# Algorithm AS 176 for calculating univariate KDE
function kde{T <: Real}(data::Vector{T},
                        window::Real = bandwidth(data),
                        npoints::Integer = 2048)
    # Determine length of data
    ndata = length(data)

    # Check that the window is a positive constant
    if window <= 0.0
        error("Window must be positive")
    end

    # Find interval that will contain almost all mass
    dlo = minimum(data) - 3 * window
    dhi = maximum(data) + 3 * window

    # Check that interval for estimation is valid
    if dlo >= dhi
        error("Interval must be specified as lower bound, upper bound")
    end

    # Set up a grid for discretized data
    grid = zeros(Float64, npoints)

    # Define some more constants
    step = (dhi - dlo) / npoints
    ainc = 1.0 / (ndata * step)
    npoints2 = fld(npoints, 2)
    hw = window / step
    fac1 = 32.0 * (atan(1.0) * hw / npoints)^2

    # Discretize the data using a histogram
    dlo1 = dlo - step
    for i in 1:ndata
        j = iround(fld(data[i] - dlo1, step))
        if j >= 1 && j <= npoints
            grid[j] = grid[j] + ainc
        end
    end

    # Transform to Fourier basis
    ft = rfft(grid)

    # Find transform of KDE by convolving grid with the
    # Fourier transform of a Gaussian kernel
    for j = 2:length(ft)
        ft[j] *= exp(-fac1 * (j-1)^2)
    end

    # Invert the Fourier transform to get the KDE
    density = irfft(ft, npoints)

    # Fix any noise that crept in
    for j in 1:npoints
        if density[j] < 0.0
            density[j] = 0.0
        end
    end

    # Expand the grid over which KDE was calculated
    x = [dlo:step:(dhi - step)]

    return UnivariateKDE(x, density)
end

# Algorithm from MASS Chapter 5 for calculating 2D KDE
function kde{S <: Real, T <: Real}(x::Vector{S},
                                   y::Vector{T};
                                   h::Real = NaN,
                                   resolution::Integer = 25)
    n = length(x)

    if length(y) != n
        error("x and y must have the same length")
    end

    if isnan(h)
        h1 = bandwidth(x)
        h2 = bandwidth(y)
    else
        h1 = h
        h2 = h
    end

    min_x, max_x = minmax(x)
    min_y, max_y = minmax(y)

    grid_x = [min_x:((max_x - min_x) / (resolution - 1)):max_x]
    grid_y = [min_y:((max_y - min_y) / (resolution - 1)):max_y]

    mx = Array(Float64, resolution, n)
    my = Array(Float64, resolution, n)
    for i in 1:resolution
        for j in 1:n
            mx[i, j] = pdf(Normal(), (grid_x[i] - x[j]) / h1)
            my[i, j] = pdf(Normal(), (grid_y[i] - y[j]) / h2)
        end
    end

    z = A_mul_Bt(mx, my)
    for i in 1:(resolution^2)
        z[i] /= (n * h1 * h2)
    end

    return BivariateKDE(grid_x, grid_y, z)
end
