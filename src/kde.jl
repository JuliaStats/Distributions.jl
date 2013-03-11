# First pass at translating Algorithm AS 176 into Julia
# Input data as a vector
# Input number of grid points at which KDE is calculated as integer
function kde(data::Vector, npoints::Integer)
    # Determine length of data
    ndata = length(data)

    # Set bandwidth to a good default for normal data
    window = 1.06 * std(data) * ndata^(-1/5)

    # Find interval that will contain most mass
    dlo = min(data) - 3 * window
    dhi = max(data) + 3 * window

    # Set up a grid for discretized data
    # In original code, FFT reuses this array
    grid = zeros(Float64, npoints)

    # Check that the window is a positive constant
    if window <= 0.0
        error("Window must be positive")
    end

    # Check that interval for estimation is valid
    if dlo >= dhi
        error("Interval must be specified as lower bound, upper bound")
    end

    # Define some more constants
    step = (dhi - dlo) / npoints
    # Why not just 1.0 / (dhi - dlo) here?
    ainc = 1.0 / (ndata * step)
    npoints2 = fld(npoints, 2)
    hw = window / step
    fac1 = 32.0 * (atan(1.0) * hw / npoints)^2

    # Discretize the data
    dlo1 = dlo - step
    for i in 1:ndata
        # Which bin does this data point belong to?
        j = fld(data[i] - dlo1, step)
        if j >= 1 && j <= npoints
            grid[j] = grid[j] + ainc
        end
    end

    # Transform to Fourier basis
    ft = fft(grid)

    # Find transform of KDE by convolving grid with FT of kernel
    # Hardcodes a Gaussian kernel
    jmax = npoints2 - 1
    smooth = Array(Complex{Float64}, npoints)
    smooth[1] = ft[1]
    rj = 0.0
    for j = 1:jmax
        rj = rj + 1.0
        fac = exp(-fac1 * rj^2)
        j1 = j + 1
        j2 = j1 + npoints2
        smooth[j1] = fac * ft[j1]
        smooth[j2] = fac * ft[j2]
    end
    smooth[npoints2 + 1] = exp(-fac1 * npoints^2) * ft[npoints2 + 1]

    # Invert the Fourier transform to get the KDE
    smooth = real(ifft(smooth))

    # Fix any noise that crept in
    for j in 1:npoints
        if smooth[j] < 0.0
            smooth[j] = 0.0
        end
    end

    # Return the grid over which KDE was calculated and KDE
    # TODO: Values of smooth oscillate improperly
    # TODO: Values of smooth are on wrong scale
    return dlo:step:(dhi - step), smooth
end
