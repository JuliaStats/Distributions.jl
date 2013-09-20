
# algorithm from:
#   J.H. Ahrens, U. Dieter (1982)
#   "Computer Generation of Poisson Deviates from Modified Normal Distributions"
#   ACM Transactions on Mathematical Software, 8(2):163-179
function rand(d::Poisson)
    μ = d.lambda
    if μ >= 10.0  # Case A

        s = sqrt(μ)
        d = 6.0*μ^2
        L = ifloor(μ-1.1484)

        # Step N
        T = randn()
        G = μ + s*T

        if G >= 0.0
            K = ifloor(G)
            # Step I
            if K >= L
                return K
            end

            # Step S
            U = rand()
            if d*U >= (μ-K)^3
                return K
            end

            # Step P
            px,py,fx,fy = procf(μ,K,s)

            # Step Q
            if fy*(1-U) <= py*exp(px-fx)
                return K
            end
        end

        while true
            # Step E
            E = Base.Random.randmtzig_exprnd()
            U = rand()
            U = 2.0*U-1.0
            T = 1.8+copysign(E,U)
            if T <= -0.6744
                continue
            end

            K = ifloor(μ + s*T)
            px,py,fx,fy = procf(μ,K,s)
            c = 0.1069/μ

            # Step H
            if c*abs(U) <= py*exp(px+E)-fy*exp(fx+E)
                return K
            end
        end
    else # Case B
        # Ahrens & Dieter use a sequential method for tabulating and looking up quantiles.
        # TODO: check which is more efficient.
        return quantile(d,rand())
    end
end
            
                
# Procedure F                    
function procf(μ,K,s)
    ω = 0.3989422804014327/s
    b1 = 0.041666666666666664/μ
    b2 = 0.3*b1^2
    c3 = 0.14285714285714285*b1*b2
    c2 = b2 - 15.0*c3
    c1 = b1 - 6.0*b2 + 45.0*c3
    c0 = 1.0 - b1 + 3.0*b2 - 15.0*c3

    if K < 10
        px = -μ
        py = μ^K/factorial(K) # replace with loopup?
    else
        δ = 0.08333333333333333/K
        δ -= 4.8*δ^3
        V = (μ-K)/K
        px = K*log1pmx(V) - δ # avoids need for table
        py = 0.3989422804014327/sqrt(K)

    end
    X = (K-μ+0.5)/s
    X2 = X^2
    fx = -0.5*X2 # missing negation in paper
    fy = ω*(((c3*X2+c2)*X2+c1)*X2+c0)
    return px,py,fx,fy
end
