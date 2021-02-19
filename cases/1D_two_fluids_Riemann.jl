#!/usr/bin/env julia

using FiniteVolumes
import LinearAlgebra
using StaticArrays

struct IsothermalTwoFluidEuler{C, R, P} <: FiniteVolumes.AbstractFlux
    c₁::C  # Sound speed of fluid 1
    ρ₁₀::R # Reference density of fluid 1
    c₂::C  # Sound speed of fluid 2
    ρ₂₀::R # Reference density of fluid 2
    p₀::P  # Reference pressure
end

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

# import Roots
# function invert_p_newton(m::IsothermalTwoFluidEuler, ρ₀, ξ, guess)
#     return Roots.find_zero((
#                             p -> ρ(m, p, ξ) - ρ₀,  # Function to cancel
#                             p -> 1.0/c²(m, p, ξ),  # Its derivative
#                            ),
#                            guess, Roots.Newton())
# end


function (f::IsothermalTwoFluidEuler)(v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p_exact(f, ρ, ξ)
    return SVector(ρ*u, ρ*u^2 + p, ρ*ξ*u) * n
end

function LinearAlgebra.eigvals(f::IsothermalTwoFluidEuler, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    c = √(c²(f, ρ, ξ))
    return SVector(u-c, u, u+c)
end

function LinearAlgebra.eigen(f::IsothermalTwoFluidEuler, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    c = √(c²(f, ρ, ξ))
    dρdξ_ = dρdξ(f, ρ, ξ)
    vals = SVector(u-c, u, u+c)
    vect =  @SMatrix [0.5  dρdξ_/ρ 0.5;
                      (u - c)/2 u*dρdξ_/ρ (c + u)/2;
                      ξ/2 (ξ*dρdξ_/ρ + 1.0) ξ/2]
    return vals, vect
end

##############################################

mesh = CartesianMesh(100)
model = IsothermalTwoFluidEuler(300.0, 1.0, 1500.0, 1000.0, 1e5)

left_state = let p=2e5, u=0.0, ξ=1.0
    r = ρ(model, p, ξ)
    SVector(r, r*u, r*u*ξ)
end
right_state = let p=1e5, u=0.0, ξ=1.0
    r = ρ(model, p, ξ)
    SVector(r, r*u, r*u*ξ)
end

v₀ = [x[1] < 0.5 ? left_state : right_state for x in cell_centers(mesh)]

Upwind()(model, mesh, v₀, (FiniteVolumes.Half(3),))
t, v = FiniteVolumes.run(model, mesh, v₀, time_step=FixedCourant(0.2), nb_time_steps=100)

using Plots
plot(mesh, [v₀ v], 1, label=["initial" "final"])
