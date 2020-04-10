import Roots

struct IsothermalTwoFluidEuler{T, Dim} <: AbstractModel
    c₁::T  # Sound speed of fluid 1
    ρ₁₀::T # Reference density of fluid 1
    c₂::T  # Sound speed of fluid 2
    ρ₂₀::T # Reference density of fluid 2
    p₀::T  # Reference pressure
end

IsothermalTwoFluidEuler{D}(args...) where D = IsothermalTwoFluidEuler{Float64, D}(args...)

function Base.convert(NT::Type{<:Number}, m::IsothermalTwoFluidEuler{T, D})  where {T, D}
    IsothermalTwoFluidEuler{NT, D}(
        convert(NT, m.c₁),
        convert(NT, m.ρ₁₀),
        convert(NT, m.c₂),
        convert(NT, m.ρ₂₀),
        convert(NT, m.p₀)
    )
end

Base.eltype(m::IsothermalTwoFluidEuler{T, D}) where {T, D} = T
nb_dims(m::IsothermalTwoFluidEuler{T, D}) where {T, D} = D
nb_vars(m::IsothermalTwoFluidEuler{T, D}) where {T, D} = 2 + D
nb_vars_supp(m::IsothermalTwoFluidEuler) = 4

w_names(m::IsothermalTwoFluidEuler{T, 1}) where T = (:p, :u, :ξ)
w_names(m::IsothermalTwoFluidEuler{T, 2}) where T = (:p, :ux, :uy, :ξ)
wsupp_names(m::IsothermalTwoFluidEuler{T, 1}) where T = (:ρ, :c, :dρdξ, :α)
wsupp_names(m::IsothermalTwoFluidEuler{T, 2}) where T = (:ρ, :c, :dρdξ, :α)

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

@inline get_pξ(m::IsothermalTwoFluidEuler{T, D}, w) where {T, D} = w[1], w[2+D]
@inline get_pρuξ(m::IsothermalTwoFluidEuler{T, 1}, w, wsupp) where {T} = w[1], wsupp[1], w[2], w[3]
@inline get_pρuξ(m::IsothermalTwoFluidEuler{T, 2}, w, wsupp) where {T} = w[1], wsupp[1], w[2], w[3], w[4]

function compute_wsupp(m::IsothermalTwoFluidEuler, w) 
    p, ξ = get_pξ(m, w)
    return SVector(ρ(m, p, ξ), sqrt(c²(m, p, ξ)), dρdξ(m, p, ξ), α(m, p, ξ))
end

########
#  1D  #
########

function rotate_state(w, wsupp, m::IsothermalTwoFluidEuler{T, 1}, rotation_matrix) where T
    w = @SVector [w[1], w[2] * rotation_matrix[1, 1], w[3]]
    return w, wsupp
end

function compute_v(m::IsothermalTwoFluidEuler{T, 1}, w, wsupp) where T
    p, ρ, u, ξ = get_pρuξ(m, w, wsupp)
    return SVector{3, T}(ρ, ρ*u, ρ*ξ)
end

function invert_v(m::IsothermalTwoFluidEuler{T, 1}, v) where T
    u = v[2]/v[1]
    ξ = v[3]/v[1]
    return SVector{3, T}(invert_p_exact(m, v[1], ξ), u, ξ)
end

function compute_w_int(m::IsothermalTwoFluidEuler{T, 1}, w_L, wsupp_L, w_R, wsupp_R) where T
    p_L, ρ_L, ux_L, ξ_L = get_pρuξ(m, w_L, wsupp_L)
    p_R, ρ_R, ux_R, ξ_R = get_pρuξ(m, w_R, wsupp_R)
    c_L = wsupp_L[2]
    c_R = wsupp_R[2]

    w_int = @SVector [
                      # p_int
                      ((ρ_L*c_L*p_L + ρ_R*c_R*p_R + ρ_L*c_L*ρ_R*c_R*(ux_L - ux_R))
                       /(ρ_L*c_L + ρ_R*c_R)),
                      # u_int
                      ((ρ_L*c_L*ux_L + ρ_R*c_R*ux_R + (p_L - p_R))
                       /(ρ_L*c_L + ρ_R*c_R)),
                      # ξ_int
                      (ξ_L + ξ_R)/2
                     ]

    return w_int, compute_wsupp(m, w_int)
end

function flux(m::IsothermalTwoFluidEuler{T, 1}, w, wsupp) where T
    p, ρ, ux, ξ = get_pρuξ(m, w, wsupp)
    return SVector(ρ*ux, ρ*ux^2 + p, ρ*ξ*ux)
end

function rotate_flux(F, m::IsothermalTwoFluidEuler{T, 1}, rotation_matrix) where T
    @SVector [F[1], F[2] * rotation_matrix[1, 1], F[3]]
end

function eigenvalues(m::IsothermalTwoFluidEuler{T, 1} , w, wsupp) where T
    u, c = w[2], wsupp[2]
    return SVector{3, T}(u-c, u, u+c)
end

function left_eigenvectors(m::IsothermalTwoFluidEuler{T, 1}, w, wsupp) where T
    p, ρ, u, ξ = get_pρuξ(m, w, wsupp)
    c, dρdξ = wsupp[2], wsupp[3]
    return @SMatrix [dρdξ*ξ/ρ + u/c + 1.0  -1.0/c  -dρdξ/ρ;
                     -ξ  0  1;
                     dρdξ*ξ/ρ - u/c + 1.0  1.0/c  -dρdξ/ρ]
end

function right_eigenvectors(m::IsothermalTwoFluidEuler{T, 1}, w, wsupp) where T
    p, ρ, u, ξ = get_pρuξ(m, w, wsupp)
    c, dρdξ = wsupp[2], wsupp[3]
    return @SMatrix [0.5  dρdξ/ρ 0.5;(u - c)/2 u*dρdξ/ρ (c + u)/2;ξ/2 (ξ*dρdξ/ρ + 1.0) ξ/2]
end

########
#  2D  #
########

function rotate_state(w, wsupp, m::IsothermalTwoFluidEuler{T, 2}, rotation_matrix) where T
    w = @SVector [w[1],
                  rotation_matrix[1, 1] * w[2] + rotation_matrix[1, 2] * w[3],
                  rotation_matrix[2, 1] * w[2] + rotation_matrix[2, 2] * w[3],
                  w[4]]
    return w, wsupp
end

function compute_v(m::IsothermalTwoFluidEuler{T, 2}, w, wsupp) where T
    p, ρ, ux, uy, ξ = get_pρuξ(m, w, wsupp)
    return SVector{4, T}(ρ, ρ*ux, ρ*uy, ρ*ξ)
end

function invert_v(m::IsothermalTwoFluidEuler{T, 2}, v) where T
    ux = v[2]/v[1]
    uy = v[3]/v[1]
    ξ = v[4]/v[1]
    return SVector{4, T}(invert_p_exact(m, v[1], ξ), ux, uy, ξ)
end

function compute_w_int(m::IsothermalTwoFluidEuler{T, 2}, w_L, wsupp_L, w_R, wsupp_R) where T
    p_L, ρ_L, ux_L, uy_L, ξ_L = get_pρuξ(m, w_L, wsupp_L)
    p_R, ρ_R, ux_R, uy_R, ξ_R = get_pρuξ(m, w_R, wsupp_R)
    c_L = wsupp_L[2]
    c_R = wsupp_R[2]

    w_int = @SVector [
                      # p_int
                      ((ρ_L*c_L*p_L + ρ_R*c_R*p_R + ρ_L*c_L*ρ_R*c_R*(ux_L - ux_R))
                       /(ρ_L*c_L + ρ_R*c_R)),
                      # ux_int
                      ((ρ_L*c_L*ux_L + ρ_R*c_R*ux_R + (p_L - p_R))
                       /(ρ_L*c_L + ρ_R*c_R)),
                      # uy_int
                      (uy_L + uy_R)/2,
                      # ξ_int
                      (ξ_L + ξ_R)/2
                     ]

    return w_int, compute_wsupp(m, w_int)
end

function flux(m::IsothermalTwoFluidEuler{T, 2}, w, wsupp) where T
    p, ρ, ux, uy, ξ = get_pρuξ(m, w, wsupp)
    return SVector(ρ*ux, ρ*ux^2 + p, ρ*ux*uy, ρ*ξ*ux)
end

function rotate_flux(F, m::IsothermalTwoFluidEuler{T, 2}, rotation_matrix) where T
    @SVector [F[1], 
              rotation_matrix[1, 1] * F[2] + rotation_matrix[1, 2] * F[3],
              rotation_matrix[2, 1] * F[2] + rotation_matrix[2, 2] * F[3],
              F[4]
             ]
end

function eigenvalues(m::IsothermalTwoFluidEuler{T, 2} , w, wsupp) where T
    u, c = w[2], wsupp[2]
    return SVector{4, T}(u-c, u, u, u+c)
end

function left_eigenvectors(m::IsothermalTwoFluidEuler{T, 2}, w, wsupp) where T
    p, ρ, ux, uy, ξ = get_pρuξ(m, w, wsupp)
    c, dρdξ = wsupp[2], wsupp[3]
    @SMatrix [dρdξ*ξ/ρ+ux/c+1.0  -1.0/c  0.0  -dρdξ/ρ;
              -uy                 0.0    1.0   0.0;
              -ξ                  0.0    0.0   1.0;
              dρdξ*ξ/ρ-ux/c+1.0   1.0/c  0.0  -dρdξ/ρ]
end

function right_eigenvectors(m::IsothermalTwoFluidEuler{T, 2}, w, wsupp) where T
    p, ρ, ux, uy, ξ = get_pρuξ(m, w, wsupp)
    c, dρdξ = wsupp[2], wsupp[3]
    @SMatrix [0.5      0.0  dρdξ/ρ       0.5;
              (ux-c)/2 0.0  ux*dρdξ/ρ    (c+ux)/2;
              uy/2     1.0  uy*dρdξ/ρ    uy/2;
              ξ/2      0.0  ξ*dρdξ/ρ+1.0 ξ/2]
end

