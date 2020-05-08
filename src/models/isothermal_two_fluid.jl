import Roots

struct IsothermalTwoFluidEuler{D, T} <: AbstractModel
    c₁::T  # Sound speed of fluid 1
    ρ₁₀::T # Reference density of fluid 1
    c₂::T  # Sound speed of fluid 2
    ρ₂₀::T # Reference density of fluid 2
    p₀::T  # Reference pressure
end

IsothermalTwoFluidEuler{D}(args...) where D = IsothermalTwoFluidEuler{D, Float64}(args...)

Base.eltype(m::IsothermalTwoFluidEuler{D, T}) where {D, T} = T
nb_dims(m::IsothermalTwoFluidEuler{D, T}) where {D, T} = D
nb_vars(m::IsothermalTwoFluidEuler{D, T}) where {D, T} = 2 + D
consvartype(m::IsothermalTwoFluidEuler{D, T}, w) where {D, T} = SVector{2 + D, T}

# EOS
ρ₁(m::IsothermalTwoFluidEuler, p) = m.ρ₁₀ + (p - m.p₀)/m.c₁^2
ρ₂(m::IsothermalTwoFluidEuler, p) = m.ρ₂₀ + (p - m.p₀)/m.c₂^2
ρ(m::IsothermalTwoFluidEuler, p, ξ) = 1/(ξ/ρ₁(m, p) + (1-ξ)/ρ₂(m, p))

c²(m::IsothermalTwoFluidEuler, p, ξ) = 1/(ρ(m, p, ξ)^2*(
                                                               ξ/(ρ₁(m, p)*m.c₁)^2
                                                        + (1 - ξ)/(ρ₂(m, p)*m.c₂)^2
                                                       )
                                         )
dρdξ(m::IsothermalTwoFluidEuler, p, ξ) = (ρ₁(m, p)-ρ₂(m, p))*ρ(m, p, ξ)^2/(ρ₁(m, p)*ρ₂(m, p))

α(m::IsothermalTwoFluidEuler, p, ξ) = ξ*ρ(m, p, ξ)/ρ₁(m, p)
ξ(m::IsothermalTwoFluidEuler, p, α) = α*ρ₁(m, p)/(α*ρ₁(m, p) + (1-α)*ρ₂(m, p))

function invert_p_exact(m::IsothermalTwoFluidEuler, ρ, ξ)
    if abs(ξ) < 10eps(typeof(ξ))
        p = m.p₀ + (ρ - m.ρ₂₀)*m.c₂^2
    elseif abs(1-ξ) < 10eps(typeof(ξ))
        p = m.p₀ + (ρ - m.ρ₁₀)*m.c₁^2
    else
        # Solve a polynomial p^2 + B p + C = 0
        B = (
             m.c₁^2*(m.ρ₁₀ - ρ*ξ)
             + m.c₂^2*(m.ρ₂₀ - (1-ξ)*ρ)
             - 2*m.p₀
            )
        C = (
             m.c₁^2*m.c₂^2*(
                            m.ρ₁₀*m.ρ₂₀
                            - m.ρ₂₀*ρ*ξ
                            - m.ρ₁₀*ρ*(1-ξ)
                           )
             - m.p₀*(
                     m.c₁^2*(m.ρ₁₀ - ρ*ξ)
                     + m.c₂^2*(m.ρ₂₀ - (1-ξ)*ρ)
                     - m.p₀
                    )
            )

        Δ = B^2 - 4*C
        p_max = (-B + sqrt(Δ))/2
        if (p_max > 0.0 && ρ₁(m, p_max) > 0.0 && ρ₂(m, p_max) > 0.0)
            return p_max
        end
        
        # Usually does not happend
        p_min = (-B - sqrt(Δ))/2
        if (p_min > 0.0 && ρ₁(m, p_min) > 0.0 && ρ₂(m, p_min) > 0.0)
            return p_min
        end

        error("Error when inverting EOS: $p_min, $p_max")
    end
end

function invert_p_newton(m::IsothermalTwoFluidEuler, ρ₀, ξ, guess)
    return Roots.find_zero((
                            p -> ρ(m, p, ξ) - ρ₀,  # Function to cancel
                            p -> 1.0/c²(m, p, ξ),  # Its derivative
                           ),
                           guess, Roots.Newton())
end

########################################

########
#  1D  #
########

w_names(m::IsothermalTwoFluidEuler{1, T}) where T = (:p, :u, :ξ, :ρ, :c, :dρdξ, :α)

full_state(m::IsothermalTwoFluidEuler{1, T}; p, u, ξ) where T = full_state(m, p, u, ξ)

function full_state(m::IsothermalTwoFluidEuler{1, T}, p, u, ξ) where T
    return (p=p, u=u, ξ=ξ,
            ρ=ρ(m, p, ξ), c=sqrt(c²(m, p, ξ)), dρdξ=dρdξ(m, p, ξ), α=α(m, p, ξ))
end

function rotate_state(w, m::IsothermalTwoFluidEuler{1, T}, rotation_matrix) where T
    return (p=w.p, u=w.u * rotation_matrix[1, 1], ξ=w.ξ, ρ=w.ρ, c=w.c, dρdξ=w.dρdξ, α=w.α)
end

function compute_v(m::IsothermalTwoFluidEuler{1, T}, w) where T
    ρ, u, ξ = w.ρ, w.u, w.ξ
    return SVector{3, eltype(m)}(ρ, ρ*u, ρ*ξ)
end

function invert_v(m::IsothermalTwoFluidEuler{1, T}, v) where T
    u = v[2]/v[1]
    ξ = v[3]/v[1]
    p = invert_p_exact(m, v[1], ξ)
    return full_state(m, p, u, ξ)
end

function compute_w_int(m::IsothermalTwoFluidEuler{1, T}, w_L, w_R) where T
    p_L, ux_L, ξ_L, ρ_L, c_L = w_L.p, w_L.u, w_L.ξ, w_L.ρ, w_L.c
    p_R, ux_R, ξ_R, ρ_R, c_R = w_R.p, w_R.u, w_R.ξ, w_R.ρ, w_R.c

    p_int = ((ρ_L*c_L*p_L + ρ_R*c_R*p_R + ρ_L*c_L*ρ_R*c_R*(ux_L - ux_R))/(ρ_L*c_L + ρ_R*c_R))
    u_int = ((ρ_L*c_L*ux_L + ρ_R*c_R*ux_R + (p_L - p_R))/(ρ_L*c_L + ρ_R*c_R))
    ξ_int = (ξ_L + ξ_R)/2
    return full_state(m, p_int, u_int, ξ_int)
end

function normal_flux(m::IsothermalTwoFluidEuler{1, T}, w) where T
    ρ, u, ξ, p = w.ρ, w.u, w.ξ, w.p
    return SVector(ρ*u, ρ*u^2 + p, ρ*ξ*u)
end

function rotate_flux(F, m::IsothermalTwoFluidEuler{1, T}, rotation_matrix) where T
    SVector{3, T}(F[1], F[2] * rotation_matrix[1, 1], F[3])
end

function eigenvalues(m::IsothermalTwoFluidEuler{1, T}, w) where T
    u, c = w.u, w.c
    return SVector{3, T}(u-c, u, u+c)
end

function left_eigenvectors(m::IsothermalTwoFluidEuler{1, T}, w) where T
    ρ, u, ξ, p = w.ρ, w.u, w.ξ, w.p
    c, dρdξ = w.c, w.dρdξ
    return @SMatrix [dρdξ*ξ/ρ + u/c + 1.0  -1.0/c  -dρdξ/ρ;
                     -ξ  0  1;
                     dρdξ*ξ/ρ - u/c + 1.0  1.0/c  -dρdξ/ρ]
end

function right_eigenvectors(m::IsothermalTwoFluidEuler{1, T}, w) where T
    ρ, u, ξ, p = w.ρ, w.u, w.ξ, w.p
    c, dρdξ = w.c, w.dρdξ
    return @SMatrix [0.5  dρdξ/ρ 0.5;
                     (u - c)/2 u*dρdξ/ρ (c + u)/2;
                     ξ/2 (ξ*dρdξ/ρ + 1.0) ξ/2]
end

########
#  2D  #
########

w_names(m::IsothermalTwoFluidEuler{2, T}) where T = (:p, :ux, :uy, :ξ, :ρ, :c, :dρdξ, :α)

full_state(m::IsothermalTwoFluidEuler{2, T}; p, ux, uy, ξ) where T = full_state(m, p, ux, uy, ξ)

function full_state(m::IsothermalTwoFluidEuler{2, T}, p, ux, uy, ξ) where T
    return (p=p, ux=ux, uy=uy, ξ=ξ,
            ρ=ρ(m, p, ξ), c=sqrt(c²(m, p, ξ)), dρdξ=dρdξ(m, p, ξ), α=α(m, p, ξ))
end

function rotate_state(w, m::IsothermalTwoFluidEuler{2, T}, rotation_matrix) where T
    return (p=w.p,
            ux=rotation_matrix[1, 1] * w[2] + rotation_matrix[1, 2] * w[3],
            uy=rotation_matrix[2, 1] * w[2] + rotation_matrix[2, 2] * w[3],
            ξ=w.ξ, ρ=w.ρ, c=w.c, dρdξ=w.dρdξ, α=w.α)
end

function compute_v(m::IsothermalTwoFluidEuler{2, T}, w) where T
    ρ, ux, uy, ξ = w.ρ, w.ux, w.uy, w.ξ
    return SVector{4, T}(ρ, ρ*ux, ρ*uy, ρ*ξ)
end

function invert_v(m::IsothermalTwoFluidEuler{2, T}, v) where T
    ux = v[2]/v[1]
    uy = v[3]/v[1]
    ξ = v[4]/v[1]
    p = invert_p_exact(m, v[1], ξ)
    return full_state(m, p, ux, uy, ξ)
end

function compute_w_int(m::IsothermalTwoFluidEuler{2, T}, w_L, w_R) where T
    p_L, ux_L, uy_L, ξ_L, ρ_L, c_L = w_L.p, w_L.ux, w_L.uy, w_L.ξ, w_L.ρ, w_L.c
    p_R, ux_R, uy_R, ξ_R, ρ_R, c_R = w_R.p, w_R.ux, w_R.uy, w_R.ξ, w_R.ρ, w_R.c

    p_int = ((ρ_L*c_L*p_L + ρ_R*c_R*p_R + ρ_L*c_L*ρ_R*c_R*(ux_L - ux_R))/(ρ_L*c_L + ρ_R*c_R))
    ux_int = ((ρ_L*c_L*ux_L + ρ_R*c_R*ux_R + (p_L - p_R))/(ρ_L*c_L + ρ_R*c_R))
    uy_int = (uy_L + uy_R)/2
    ξ_int = (ξ_L + ξ_R)/2
    return full_state(m, p_int, ux_int, uy_int, ξ_int)
end

function normal_flux(m::IsothermalTwoFluidEuler{2, T}, w) where T
    ρ, ux, uy, ξ, p = w.ρ, w.ux, w.uy, w.ξ, w.p
    return SVector{4, T}(ρ*ux, ρ*ux^2 + p, ρ*ux*uy, ρ*ξ*ux)
end

function rotate_flux(F, m::IsothermalTwoFluidEuler{2, T}, rotation_matrix) where T
    SVector{4, T}(F[1], 
                  rotation_matrix[1, 1] * F[2] + rotation_matrix[1, 2] * F[3],
                  rotation_matrix[2, 1] * F[2] + rotation_matrix[2, 2] * F[3],
                  F[4]
                 )
end

function eigenvalues(m::IsothermalTwoFluidEuler{2, T}, w) where T
    u, c = w.ux, w.c
    return SVector{4, T}(u-c, u, u, u+c)
end

function left_eigenvectors(m::IsothermalTwoFluidEuler{2, T}, w) where T
    ρ, ux, uy, ξ, p = w.ρ, w.ux, w.uy, w.ξ, w.p
    c, dρdξ = w.c, w.dρdξ
    @SMatrix [dρdξ*ξ/ρ+ux/c+1.0  -1.0/c  0.0  -dρdξ/ρ;
              -uy                 0.0    1.0   0.0;
              -ξ                  0.0    0.0   1.0;
              dρdξ*ξ/ρ-ux/c+1.0   1.0/c  0.0  -dρdξ/ρ]
end

function right_eigenvectors(m::IsothermalTwoFluidEuler{2, T}, w) where T
    ρ, ux, uy, ξ, p = w.ρ, w.ux, w.uy, w.ξ, w.p
    c, dρdξ = w.c, w.dρdξ
    @SMatrix [0.5      0.0  dρdξ/ρ       0.5;
              (ux-c)/2 0.0  ux*dρdξ/ρ    (c+ux)/2;
              uy/2     1.0  uy*dρdξ/ρ    uy/2;
              ξ/2      0.0  ξ*dρdξ/ρ+1.0 ξ/2]
end

