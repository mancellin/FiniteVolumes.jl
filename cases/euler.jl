#!/usr/bin/env julia

using FiniteVolumes
import LinearAlgebra
using StaticArrays

# EOS
struct IsothermalStiffenedGas{C, R, P}
    c::C   # Sound speed
    ρ₀::R  # Reference density
    p₀::P  # Reference pressure
end

ρ(m::IsothermalStiffenedGas, p) = m.ρ₀ + (p - m.p₀)/m.c^2
ρ²c²(m::IsothermalStiffenedGas, p) = ρ(m, p)^2*m.c^2

struct TwoFluidMixture{F1, F2}
    eos1::F1
    eos2::F2
end

ρ(m::TwoFluidMixture, p, ξ) = 1/(ξ/ρ(m.eos1, p) + (1-ξ)/ρ(m.eos2, p))
ρ²c²(m::TwoFluidMixture, p, ξ) = 1/(ξ/ρ²c²(m.eos1, p) + (1 - ξ)/ρ²c²(m.eos2, p))
dρdξ(m::TwoFluidMixture, p, ξ) = (ρ₁ = ρ(m.eos1, p); ρ₂ = ρ(m.eos2, p); (ρ₂-ρ₁)*ρ(m, p, ξ)^2/(ρ₁*ρ₂))
α(m::TwoFluidMixture, p, ξ) = ξ*ρ(m, p, ξ)/ρ(m.eos1, p)
ξ(m::TwoFluidMixture, p, α) = α*ρ(m.eos1, p)/(α*ρ(m.eos1, p) + (1-α)*ρ(m.eos2, p))

function invert_p_exact(m::TwoFluidMixture, ρ_, ξ)
    if abs(ξ) < 10eps(typeof(ξ))
        p = m.eos2.p₀ + (ρ_ - m.eos2.ρ₀)*m.eos2.c^2
    elseif abs(1-ξ) < 10eps(typeof(ξ))
        p = m.eos1.p₀ + (ρ_ - m.eos1.ρ₀)*m.eos1.c^2
    else
        # Solve a polynomial p^2 + B p + C = 0
        c₁ = m.eos1.c; ρ₁₀ = m.eos1.ρ₀; p₀₁ = m.eos1.p₀
        c₂ = m.eos2.c; ρ₂₀ = m.eos2.ρ₀; p₀₂ = m.eos2.p₀
        B = c₁^2*(ρ₁₀ - ρ_*ξ) - p₀₁ + c₂^2*(ρ₂₀ - (1-ξ)*ρ_) - p₀₂
        C = (c₁^2*c₂^2*(ρ₁₀*ρ₂₀ - ρ₂₀*ρ_*ξ - ρ₁₀*ρ_*(1-ξ))
             - p₀₁*c₁^2*(ρ₁₀ - ρ_*ξ) - p₀₂*c₂^2*(ρ₂₀ - (1-ξ)*ρ_) + p₀₁*p₀₂)
        Δ = B^2 - 4*C
        p_max = (-B + sqrt(Δ))/2
        if (p_max > 0.0 && ρ(m.eos1, p_max) > 0.0 && ρ(m.eos2, p_max) > 0.0)
            return p_max
        end
        # Usually does not happend
        p_min = (-B - sqrt(Δ))/2
        if (p_min > 0.0 && ρ(m.eos1, p_min) > 0.0 && ρ(m.eos2, p_min) > 0.0)
            return p_min
        end
        error("Error when inverting EOS: $p_min, $p_max")
    end
end

import Roots
function invert_p_newton(m::TwoFluidMixture, ρ₀, ξ, guess)
    f, df = (p -> ρ(m, p, ξ) - ρ₀, p -> ρ(m, p, ξ)^2/ρ²c²(m, p, ξ))
    return Roots.find_zero((f, df), guess, Roots.Newton())
end

##############################################

const model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1e5), IsothermalStiffenedGas(1500.0, 1000.0, 1e5))

@assert invert_p_exact(model, ρ(model, 1e5, 0.0), 0.0) == 1e5
@assert invert_p_exact(model, ρ(model, 1e5, 0.5), 0.5) == 1e5
@assert invert_p_exact(model, ρ(model, 1e5, 1.0), 1.0) == 1e5
@assert invert_p_exact(model, ρ(model, 1.5e5, 0.0), 0.0) ≈ 1.5e5
@assert invert_p_exact(model, ρ(model, 1.5e5, 0.5), 0.5) ≈ 1.5e5
@assert invert_p_exact(model, ρ(model, 1.5e5, 1.0), 1.0) ≈ 1.5e5

##############################################

struct EulerFlux <: FiniteVolumes.AbstractFlux end

function (::EulerFlux)(v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p_exact(model, ρ, ξ)
    return SVector(ρ*u, ρ*u^2 + p, ρ*ξ*u) * n
end

function LinearAlgebra.eigvals(::EulerFlux, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p_exact(model, ρ, ξ)
    c = √(ρ²c²(model, p, ξ))/ρ
    return SVector(u-c, u, u+c)
end

function LinearAlgebra.eigen(::EulerFlux, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p_exact(model, ρ, ξ)
    c = √(ρ²c²(model, p, ξ))/ρ
    dρdξ_ = dρdξ(model, p, ξ)
    vals = SVector(u-c, u, u+c)
    vect =  @SMatrix [0.5         dρdξ_/ρ             0.5;
                      (u - c)/2   u*dρdξ_/ρ           (c + u)/2;
                      ξ/2         (ξ*dρdξ_/ρ + 1.0)   ξ/2]
    return vals, vect
end

##############################################

mesh = CartesianMesh(100)

left_state = let p=2e5, u=0.0, ξ=1.0
    r = ρ(model, p, ξ)
    SVector(r, r*u, r*ξ)
end
right_state = let p=1e5, u=0.0, ξ=1.0
    r = ρ(model, p, ξ)
    SVector(r, r*u, r*ξ)
end

v₀ = [x[1] < 0.5 ? left_state : right_state for x in cell_centers(mesh)]

t, v = FiniteVolumes.run(EulerFlux(), mesh, v₀, time_step=FixedCourant(0.2), nb_time_steps=100)

using Plots
plot(mesh, [v₀ v], 1, label=["initial" "final"])
