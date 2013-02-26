# Written in Julia by Andreas Noack Jensen
# January 2013
#
# Translation of Fortran file tvpack.f authored by
#
# Alan Genz
# Department of Mathematics
# Washington State University
# Pullman, WA 99164-3113
# Email : alangenz@wsu.edu
#
# Original source available from
# http://www.math.wsu.edu/faculty/genz/software/fort77/tvpack.f


function tvtcdf(nu::Int, h::Vector{Float64}, r::Vector{Float64})
	pt = 0.5 * pi
	h1 = h[1]
	h2 = h[2]
	h3 = h[3]
	r12 = r[1]
	r13 = r[2]
	r23 = r[3]
	
	# Sort R's and check for special cases
	
	if abs(r12) > abs(r13)
	   h2 = h3
	   h3 = h[2]
	   r12 = r13
	   r13 = r[1]
	end
	if abs(r13) > abs(r23)
	   h1 = h2
	   h2 = h[1]
	   r23 = r13
	   r13 = r[3]
	end
	tvt = 0.0
	if abs(h1) + abs(h2) + abs(h3) < eps()
	   	tvt = (1.0 + (asin(r12) + asin(r13) + asin(r23)) / pt) * 0.125
	elseif nu < 1 && abs(r12) + abs(r13) < eps()
	   	tvt = cdf(Normal(), h1) * bvtcdf(nu, h2, h3, r23)
	elseif nu < 1 && abs(r13) + abs(r23) < eps()
	   	tvt = cdf(Normal(), h3) * bvtcdf(nu, h1, h2, r12)
	elseif nu < 1 && abs(r12) + abs(r23) < eps()
	   	tvt = cdf(Normal(), h2) * bvtcdf(nu, h1, h3, r13)
	elseif 1.0 - r23 < eps()
	   	tvt = bvtnorm(nu, h1, min(h2, h3), r12)
	elseif r23 + 1.0 < eps()
	   	if h2 > -h3 tvt = bvtcdf(nu, h1, h2, r12) - bvtcdf(nu, h1, -h3, r12) end
	else
	
	   	# Compute singular TVT value
	
	   	if nu < 1
	      	tvt = bvtcdf(nu, h2, h3, r23) * cdf(Normal(), h1)
	   	elseif r23 > 0
	      	tvt = bvtcdf( nu, h1, min( h2, h3 ), 0.0)
	   	elseif h2 > -h3
	      	tvt = bvtcdf(nu, h1, h2, 0.0) - bvtcdf(nu, h1, -h3, 0.0)
	   	end
	   	# Use numerical integration to compute probability
		
	   	rua = asin(r12)
	   	rub = asin(r13)
	   	ar = asin(r23)
	   	ruc = pt * sign(ar) - ar
	   	tvt += adonet(x->tvtmfn(x, rua, rub, nu, h1, h2, h3, r23, ar, ruc), 0.0, 1.0, sqrt(eps())) / (4.0*pt)
	end
	return max(0.0, min(tvt, 1.0)) 
end

#Computes Plackett formula integrands
function tvtmfn(x::Float64, rua::Float64, rub::Float64, nu::Int, h1::Float64, h2::Float64, h3::Float64, r23::Float64, ar::Float64, ruc::Float64)
	retval = 0.0
	r12, rr2 = sincs(rua * x)
	r13, rr3 = sincs(rub * x)
	if abs(rua) > 0
     	retval += rua * pntgnd(nu, h1, h2, h3, r13, r23, r12, rr2)
	end
	if abs(rub) > 0
     	retval += rub * pntgnd(nu, h1, h3, h2, r12, r23, r13, rr3)
	end
	if nu > 0
   		r, rr = sincs(ar + ruc * x)
   		retval -= ruc * pntgnd(nu, h2, h3, h1, 0.0, 0.0, r, rr)
	end
	return retval
end

# Computes SIN(X), COS(X)^2, with series approx. for |X| near PI/2
function sincs(x::Float64)
	ee = (0.5 * pi - abs(x))^2
	if ee < 5e-5
   		sx = abs(1.0 - ee * (1.0 - ee / 12.0) * 0.5) * sign(x)
   		cs = ee * (1.0 - ee * (1.0 - 2.0 * ee / 15.0) / 3.0)
	else
   		sx = sin(x)
   		cs = 1.0 - sx*sx
	end
	return sx, cs
end

# Computes Plackett formula integrand
function pntgnd(nu::Int, ba::Float64, bb::Float64, bc::Float64, ra::Float64, rb::Float64, r::Float64, rr::Float64)
	retval = 0.0
	dt = rr * (rr - (ra - rb)^2 - 2.0 * ra * rb * (1.0 - r))
	if dt > 0
   		bt = (bc * rr + ba * (r * rb - ra) + bb * (r * ra - rb)) / sqrt(dt)
   		ft = (ba - r * bb)^2 / rr + bb * bb
   		if nu < 1
      		if bt > -10 && ft < 100
         		retval = exp(-ft * 0.5)
         		if bt < 10; retval *= cdf(Normal(), bt) end
      		end
   		else
      		ft = sqrt(1.0 + ft / nu)
      		retval = tcdf(nu, bt / ft) / ft^nu
   		end
	end
	return retval
end

# # One Dimensional Globally Adaptive Integration Function

function adonet(f::Function, a::Float64, b::Float64, tol::Float64)
	ai = Array(Float64, 100)
	bi = Array(Float64, 100)
	ei = Array(Float64, 100)
	fi = Array(Float64, 100)
	ai[1] = a
	bi[1] = b
	err = 1.0
	ip = 1
	im = 1
	fin = 0.0
	while 4.0err > tol && im < 100
	   	im += 1
	   	bi[im] = bi[ip]
	   	ai[im] = (ai[ip] + bi[ip]) * 0.5
	   	bi[ip] = ai[im]
	   	fi[ip], ei[ip] = krnrdt(ai[ip], bi[ip], f)
		fi[im], ei[im] = krnrdt(ai[im], bi[im], f)
	   	err = 0.0
	   	fin = 0.0
	   	for i = 1:im
	      	if ei[i] > ei[ip]; ip = i; end
	      	fin += fi[i]
	      	err += ei[i]^2
	   	end
	   	err = sqrt(err)
	end
	return fin
end

# # Kronrod Rule

const kronrod_wg = [0.2729250867779007e+00,
					0.5566856711617449e-01,
					0.1255803694649048e+00,
					0.1862902109277352e+00,
					0.2331937645919914e+00,
					0.2628045445102478e+00]

const kronrod_xgk = [0.0000000000000000e+00,
					 0.9963696138895427e+00,
					 0.9782286581460570e+00,
					 0.9416771085780681e+00,
					 0.8870625997680953e+00,
					 0.8160574566562211e+00,
					 0.7301520055740492e+00,
					 0.6305995201619651e+00,
					 0.5190961292068118e+00,
					 0.3979441409523776e+00,
					 0.2695431559523450e+00,
					 0.1361130007993617e+00]

const kronrod_wgk = [0.1365777947111183e+00,
					 0.9765441045961290e-02,
					 0.2715655468210443e-01,
					 0.4582937856442671e-01,
					 0.6309742475037484e-01,
					 0.7866457193222764e-01,
					 0.9295309859690074e-01,
					 0.1058720744813894e+00,
					 0.1167395024610472e+00,
					 0.1251587991003195e+00,
					 0.1312806842298057e+00,
					 0.1351935727998845e+00]

function krnrdt(a::Float64, b::Float64, f::Function)
	wid = (b - a) * 0.5
	cen = (b + a) * 0.5
	fc = f(cen)
	resg = fc * kronrod_wg[1]
	resk = fc * kronrod_wgk[1]
	for j = 1:11
	   	t = wid * kronrod_xgk[j + 1] 
	   	fc = f(cen - t) + f(cen + t)
	   	resk += kronrod_wgk[j + 1] * fc
	   	if mod(j, 2) == 0; resg += kronrod_wg[div(j,2) + 1] * fc; end
	end
	retval = wid * resk
	err = abs(wid * (resk - resg))
	return retval, err
end

# # Student t Distribution Function

function tcdf(nu::Int, t::Float64)
	if nu < 1
   		studnt = cdf(Normal(), t)
	elseif nu == 1
   		studnt = (1.0 + 2.0atan(t) / pi) * 0.5
	elseif nu == 2
   		studnt = (1.0 + t / sqrt(2.0 + t*t)) * 0.5
	else 
   		tt = t*t
   		cssthe = 1.0 / (1.0 + tt / nu)
   		polyn = 1.0
   		for j = nu-2:-2:2
    	  	polyn = 1.0 + (j - 1) * cssthe * polyn / j
   		end
   		if mod(nu, 2) == 1
    	  	rn = nu
    	  	ts = t / sqrt(rn)
    	  	studnt = (1.0 + 2.0*(atan(ts) + ts * cssthe * polyn) / pi) * 0.5
   		else
    	  	snthe = t / sqrt(nu + tt)
    	  	studnt = (1.0 + snthe * polyn) * 0.5
   		end
   		studnt = max(0.0, min(studnt, 1.0))
	end
	return studnt
end

function bvtcdf(nu::Int, dh::Float64, dk::Float64, r::Float64)
	if nu < 1
		return bvnuppercdf(-dh, -dk, r)
	elseif 1.0 - r < eps()
		return tcdf(nu, min(dh, dk))
	elseif r + 1.0 < eps()
		if dh > -dk
			return tcdf(nu, dh) - tcdf(nu, -dk)
		else
			return 0.0
		end
	else
		snu = sqrt(nu)
		ors = 1.0 - r * r
		hrk = dh - r * dk
		krh = dk - r * dh
		if abs(hrk) + ors > 0
			xnhk = hrk^2 / (hrk^2 + ors * (nu + dk^2))
			xnkh = krh^2 / (krh^2 + ors * (nu + dh^2))
		else
			xnhk = 0.0
			xnkh = 0.0
		end
		hs = sign(hrk)
		ks = sign(krh)
		if mod(nu, 2) == 0
			bvt = atan2(sqrt(ors), -r) / (2.0pi)
			gmph = dh / sqrt(16.0 * (nu + dh^2))
			gmpk = dk / sqrt(16.0 * (nu + dk^2))
			btnckh = 2.0atan2(sqrt(xnkh), sqrt(1.0 - xnkh)) / pi
            btpdkh = 2.0sqrt(xnkh * (1.0 - xnkh)) / pi
            btnchk = 2.0atan2(sqrt(xnhk), sqrt(1.0 - xnhk)) / pi
            btpdhk = 2.0sqrt(xnhk * (1.0 - xnhk)) / pi
            for j = 1:div(nu, 2)
               	bvt += gmph * (1.0 + ks * btnckh)
               	bvt += gmpk * (1.0 + hs * btnchk)
               	btnckh += btpdkh
               	btpdkh *= 2.0j * (1.0 - xnkh) / (2j + 1)
               	btnchk += btpdhk
               	btpdhk *= 2.0j * (1.0 - xnhk) / (2j + 1)
               	gmph *= (2j - 1) / (2j * (1.0 + dh * dh / nu)) 
               	gmpk *= (2j - 1) / (2j * (1.0 + dk * dk / nu)) 
            end
        else
            qhrk = sqrt(dh * dh + dk * dk - 2.0r*dh*dk + nu*ors)  
            hkrn = dh*dk + r*nu
            hkn = dh*dk - nu
            hpk = dh + dk
            bvt = atan2(-snu * (hkn * qhrk + hpk * hkrn), hkn * hkrn - nu * hpk * qhrk) / (2.0pi)
            if bvt < -eps() 
            	bvt += 1.0
            end
            gmph = dh / (2.0pi * snu * (1.0 + dh * dh / nu))
            gmpk = dk / (2.0pi * snu * (1.0 + dk * dk / nu))
            btnckh = sqrt(xnkh)
            btpdkh = btnckh
            btnchk = sqrt(xnhk)
            btpdhk = btnchk
            for j = 1:div(nu - 1, 2)
               	bvt += gmph * (1.0 + ks * btnckh)
               	bvt += gmpk * (1.0 + hs * btnchk)
               	btpdkh *= (2j - 1) * (1.0 - xnkh) / (2*j)  
               	btnckh += btpdkh  
               	btpdhk *= (2j - 1) * (1.0 - xnhk) / (2*j)  
               	btnchk += btpdhk  
               	gmph *= 2.0j / ((2j + 1) * (1.0 + dh * dh / nu ))
               	gmpk *= 2.0j / ((2j + 1) * (1.0 + dk * dk / nu ))
            end
        end
        return bvt 
    end
end

# This function is based on the method described by 
#     Drezner, Z and G.O. Wesolowsky, (1989),
#     On the computation of the bivariate normal integral,
#     Journal of Statist. Comput. Simul. 35, pp. 101-107,
# with major modifications for double precision, and for |R| close to 1.

const bvncdf_w_array = [0.1713244923791705e+00 0.4717533638651177e-01 0.1761400713915212e-01;
     					0.3607615730481384e+00 0.1069393259953183e+00 0.4060142980038694e-01;
     					0.4679139345726904e+00 0.1600783285433464e+00 0.6267204833410906e-01;
     					0.0 				   0.2031674267230659e+00 0.8327674157670475e-01;
     					0.0					   0.2334925365383547e+00 0.1019301198172404e+00;
     					0.0					   0.2491470458134029e+00 0.1181945319615184e+00;
     					0.0					   0.0					  0.1316886384491766e+00;
     					0.0					   0.0					  0.1420961093183821e+00;
     					0.0					   0.0					  0.1491729864726037e+00;
     					0.0					   0.0					  0.1527533871307259e+00]

const bvncdf_x_array = [-0.9324695142031522e+00 -0.9815606342467191e+00 -0.9931285991850949e+00;
						-0.6612093864662647e+00 -0.9041172563704750e+00 -0.9639719272779138e+00;
						-0.2386191860831970e+00 -0.7699026741943050e+00 -0.9122344282513259e+00;
						 0.0 				    -0.5873179542866171e+00 -0.8391169718222188e+00;
						 0.0 				    -0.3678314989981802e+00 -0.7463319064601508e+00;
						 0.0 				    -0.1252334085114692e+00 -0.6360536807265150e+00;
						 0.0 				    0.0 				    -0.5108670019508271e+00;
						 0.0 				    0.0 				    -0.3737060887154196e+00;
						 0.0 				    0.0 				    -0.2277858511416451e+00;
						 0.0 				    0.0 				    -0.7652652113349733e-01]

function bvnuppercdf(dh::Float64, dk::Float64, r::Float64)
	if abs(r) < 0.3
	   ng = 1
	   lg = 3
	elseif abs(r) < 0.75
	   ng = 2
	   lg = 6
	else
	   ng = 3
	   lg = 10
	end
	h = dh
	k = dk 
	hk = h*k
	bvn = 0.0
	if abs(r) < 0.925
	   	if abs(r) > 0
	      	hs = (h * h + k * k) * 0.5
	      	asr = asin(r)
	      	for i = 1:lg
	         	for j = -1:2:1
	            	sn = sin(asr * (j * bvncdf_x_array[i, ng] + 1.0) * 0.5)
	            	bvn += bvncdf_w_array[i, ng] * exp((sn * hk - hs) / (1.0 - sn*sn))
	        	end
	      	end
	      	bvn *= asr / (4.0pi)
	   	end
	   	bvn += cdf(Normal(), -h) * cdf(Normal(), -k)
	else
	   	if r < 0
	      	k = -k
	      	hk = -hk
	   	end	   
	   	if abs(r) < 1
	      	as = (1.0 - r) * (1.0 + r)
	      	a = sqrt(as)
	      	bs = (h - k)^2
	      	c = (4.0 - hk) * 0.125
	      	d = (12.0 - hk) * 0.0625
	      	asr = -(bs / as + hk) * 0.5
	      	if ( asr .gt. -100 ) 
	      		bvn = a * exp(asr) * (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 + c * d * as * as / 5.0)
	      	end
	      	if -hk < 100
	         	b = sqrt(bs)
	         	bvn -= exp(-hk * 0.5) * sqrt(2.0pi) * cdf(Normal(), -b / a) * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
	      	end
	     	a /= 2.0
		    for i = 1:lg
	         	for j = -1:2:1
	            	xs = (a * (j*bvncdf_x_array[i, ng] + 1.0))^2
	            	rs = sqrt(1.0 - xs)
	            	asr = -(bs / xs + hk) * 0.5
	            	if asr > -100
	               		bvn += a * bvncdf_w_array[i, ng] * exp(asr) * (exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs - (1.0 + c * xs * (1.0 + d * xs)))
	            	end
	         	end
	        end
	      	bvn /= -2.0pi
	   	end
	   	if r > 0
	      	bvn += cdf(Normal(), -max(h, k))
	   	else
	      	bvn = -bvn
	      	if k > h
	      		bvn += cdf(Normal(), k) - cdf(Normal(), h)
	      	end
		end
	end
	return bvn
end