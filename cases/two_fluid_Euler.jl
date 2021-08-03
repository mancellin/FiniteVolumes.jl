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
        error("Error when inverting EOS: $ρ_, $ξ, $p_min, $p_max")
    end
end

import Roots
function invert_p_newton(m::TwoFluidMixture, ρ₀, ξ, guess=m.eos1.p₀)
    f, df = (p -> ρ(m, p, ξ) - ρ₀, p -> ρ(m, p, ξ)^2/ρ²c²(m, p, ξ))
    return Roots.find_zero((f, df), guess, Roots.Newton())
end

const invert_p = invert_p_newton

# p_int = ((ρ_L*c_L*p_L + ρ_R*c_R*p_R + ρ_L*c_L*ρ_R*c_R*(ux_L - ux_R))/(ρ_L*c_L + ρ_R*c_R))
# ux_int = ((ρ_L*c_L*ux_L + ρ_R*c_R*ux_R + (p_L - p_R))/(ρ_L*c_L + ρ_R*c_R))
# uy_int = (uy_L + uy_R)/2
# ξ_int = (ξ_L + ξ_R)/2

# (ρ²c²(model, 1.0, 0.0)/ρ(model, 1.0, 0.0)) / (ρ²c²(model, 1.0, 1.0)/ρ(model, 1.0, 1.0))

##############################################

using Test
function test_eos()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1.0), IsothermalStiffenedGas(1500.0, 1000.0, 1.0))
    @test invert_p_newton(model, ρ(model, 1.0, 0.0), 0.0) == 1.0
    @test invert_p_newton(model, ρ(model, 1.0, 0.5), 0.5) == 1.0
    @test invert_p_newton(model, ρ(model, 1.0, 1.0), 1.0) == 1.0
    @test invert_p_newton(model, ρ(model, 1.5, 0.0), 0.0) ≈ 1.5 atol=1e-5
    @test invert_p_newton(model, ρ(model, 1.5, 0.5), 0.5) ≈ 1.5 atol=1e-5
    @test invert_p_newton(model, ρ(model, 1.5, 1.0), 1.0) ≈ 1.5 atol=1e-5
end
test_eos()

##############################################

struct EulerFlux{EOS} <: FiniteVolumes.AbstractFlux
    eos::EOS
end

########
#  1D  #
########

function (f::EulerFlux)(v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    return SVector(ρ*u, ρ*u^2 + p, ρ*ξ*u)
end

function LinearAlgebra.eigvals(f::EulerFlux, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    c = √(ρ²c²(f.eos, p, ξ))/ρ
    return SVector(u-c, u, u+c)
end

function LinearAlgebra.eigen(f::EulerFlux, v, n::Number)
    ρ = v[1]
    u = v[2]/v[1] * n
    ξ = v[3]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    c = √(ρ²c²(f.eos, p, ξ))/ρ
    dρdξ_ = dρdξ(f.eos, p, ξ)
    vals = SVector(u-c, u, u+c)
    vect =  @SMatrix [0.5         dρdξ_/ρ             0.5;
                      (u - c)/2   u*dρdξ_/ρ           (c + u)/2;
                      ξ/2         (ξ*dρdξ_/ρ - 1.0)   ξ/2]
    return vals, vect
end
# @SMatrix [dρdξ*ξ/ρ + u/c + 1.0  -1.0/c  -dρdξ/ρ;
#           -ξ  0  1;
#           dρdξ*ξ/ρ - u/c + 1.0  1.0/c  -dρdξ/ρ]

using ForwardDiff
function test_jacobian()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1.0),
                            IsothermalStiffenedGas(1500.0, 1000.0, 1.0))
    flux = EulerFlux(model)

    n = 1.0
    for v0 in [SVector(1.0, 0.0, 1.0), SVector(1000.0, 0.0, 0.0),
               SVector(1.0, 10.0, 1.0), SVector(1000.0, 10.0, 0.0),
               SVector(1.0, -10.0, 1.0), SVector(1000.0, -10.0, 0.0),
              ]
        λ, R = LinearAlgebra.eigen(ForwardDiff.jacobian(v -> flux(v, n), v0))
        λ1, R1 = LinearAlgebra.eigen(flux, v0, n)
        for i in 1:3
            @assert R[:, i]/R[1, i] ≈ R1[:, i]/R1[1, i]
        end
    end
end
test_jacobian()

########
#  2D  #
########

function (f::EulerFlux)(v, n::SVector{2})
    ρ = v[1]
    ux, uy = v[2]/v[1], v[3]/v[1]
    ξ = v[4]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    R = SMatrix{2, 2}(n[1], -n[2], n[2], n[1])
    u_local = R * SVector(ux, uy)
    ϕu_local = SVector(ρ*u_local[1]^2 + p, ρ*u_local[1]*u_local[2]) 
    ϕu = R' * ϕu_local
    return SVector(ρ*u_local[1], ϕu[1], ϕu[2], ρ*ξ*u_local[1])
end

function LinearAlgebra.eigvals(f::EulerFlux, v, n::SVector{2})
    ρ = v[1]
    ux, uy = v[2]/v[1], v[3]/v[1]
    R = SMatrix{2, 2}(n[1], -n[2], n[2], n[1])
    u_local = R * SVector(ux, uy)
    ξ = v[4]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    c = √(ρ²c²(f.eos, p, ξ))/ρ
    return SVector(u_local[1]-c, u_local[1], u_local[1], u_local[1]+c)
end

function RR(n::SVector{2, T}) where T
    z = zero(T)
    i = oneunit(T)
    SMatrix{4, 4}(i, z, z, z,
                  z, n[1], -n[2], z,
                  z, n[2], n[1], z,
                  z, z, z, i)
end

function LinearAlgebra.eigen(f::EulerFlux, v, n::SVector{2})
    ρ = v[1]
    ux, uy = v[2]/v[1], v[3]/v[1]
    R = SMatrix{2, 2}(n[1], -n[2], n[2], n[1])
    u_local = R * SVector(ux, uy)
    ux, uy = u_local
    ξ = v[4]/v[1]
    p = invert_p(f.eos, ρ, ξ)
    c = √(ρ²c²(f.eos, p, ξ))/ρ
    dρdξ_ = dρdξ(f.eos, p, ξ)
    vals = SVector(ux-c, ux, ux, ux+c)
    vect = @SMatrix [0.5      0.0  dρdξ_/ρ       0.5;
                     (ux-c)/2 0.0  ux*dρdξ_/ρ    (c+ux)/2;
                     uy/2     1.0  uy*dρdξ_/ρ    uy/2;
                     ξ/2      0.0  ξ*dρdξ_/ρ-1.0 ξ/2]
    return vals, RR(n)' * vect * RR(n)
end

    # @SMatrix [dρdξ*ξ/ρ+ux/c+1.0  -1.0/c  0.0  -dρdξ/ρ;
    #           -uy                 0.0    1.0   0.0;
    #           -ξ                  0.0    0.0   1.0;
    #           dρdξ*ξ/ρ-ux/c+1.0   1.0/c  0.0  -dρdξ/ρ]


###################################
struct LUpwind <: FiniteVolumes.NumericalFlux end
###################################

function two_point_l_upwind(flux, v₁, v₂, n)  # l-upwind FVCF
    v_mean = (v₁ + v₂)/2
    λ = LinearAlgebra.eigvals(flux, v_mean, n)

    F₁ = flux(v₁, n)
    F₂ = flux(v₂, n)

    λ₁, R₁ = LinearAlgebra.eigen(flux, v₁, n)
    λ₂, R₂ = LinearAlgebra.eigen(flux, v₂, n)

    L₁ = inv(R₁)
    L₂ = inv(R₂)

    L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
    L_flux_upwind = ifelse.(λ .> 0.0, L₁ * F₁, L₂ * F₂)

    ϕ = L_upwind \ L_flux_upwind
    return ϕ
end

function (::LUpwind)(flux, mesh, v, i_face)  # l-upwind FVCF
    n = FiniteVolumes.normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = FiniteVolumes.cells_next_to_inner_face(mesh, i_face)
    return two_point_l_upwind(flux, v[i_cell_1], v[i_cell_2], n)
end

##############################################

using Plots

function test_pure_riemann_1D()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1e5), IsothermalStiffenedGas(1500.0, 1000.0, 1e5))
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

    t, v = FiniteVolumes.run(EulerFlux(model), mesh, v₀, time_step=FixedCourant(0.2), nb_time_steps=100)

    plot(mesh, [v₀ v], 1, label=["initial" "final"])
end

##############################################

function test_pure_riemann_2D()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1e5), IsothermalStiffenedGas(1500.0, 1000.0, 1e5))
    mesh = CartesianMesh(40, 40)

    left_state = let p=2e5, ux=0.0, uy=0.0, ξ=1.0
        r = ρ(model, p, ξ)
        SVector(r, r*ux, r*uy, r*ξ)
    end
    right_state = let p=1e5, ux=0.0, uy=0.0, ξ=1.0
        r = ρ(model, p, ξ)
        SVector(r, r*ux, r*uy, r*ξ)
    end

    v₀ = [x[1] + 2*x[2] < 1.5 ? left_state : right_state for x in cell_centers(mesh)]
    # FiniteVolumes.div(EulerFlux(), mesh, v₀)

    t, v = FiniteVolumes.run(EulerFlux(model), mesh, v₀, time_step=FixedCourant(0.2), nb_time_steps=80)

    plot(mesh, v, 1, label="final")
end

##############################################

function test_advection_1D()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1.0),
                            IsothermalStiffenedGas(1500.0, 1000.0, 1.0))
    mesh = PeriodicCartesianMesh(10)

    initial_state(ξ) = let p=1.0, u=1.0
        r = ρ(model, p, ξ)
        SVector(r, r*u, r*ξ)
    end
    v₀ = [x < 0.3 ? initial_state(1.0) :
          x < 0.8 ? initial_state(0.0) :
                    initial_state(1.0)
          for x in cell_centers(mesh)]

    time_step = FixedCourant(0.2)
    nb_time_steps = 10
    numerical_flux = LUpwind()
    t, v = FiniteVolumes.run(EulerFlux(model), mesh, v₀; time_step, nb_time_steps, numerical_flux)

    ξ_initial = (vi -> vi[3]/vi[1]).(v₀)
    ξ_final = (vi -> vi[3]/vi[1]).(v)
    plot(mesh, [ξ_initial ξ_final], label=["ξ Initial" "ξ Final"])

    p_final = (vi -> invert_p_exact(model, vi[1], vi[3]/vi[1])).(v)
    plot!(mesh, p_final, label="p final")
end

##############################################

struct PistonBC <: FiniteVolumes.BoundaryCondition end

function (::PistonBC)(flux, mesh, v, i_face)
    n = FiniteVolumes.normal_vector(mesh, i_face)
    i_cell_1 = FiniteVolumes.cell_next_to_boundary_face(mesh, i_face)
    if i_cell_1 == CartesianIndex(1)
        return flux(v[i_cell_1], n)
    else
        v₂ = SVector(v[i_cell_1][1], -v[i_cell_1][2], v[i_cell_1][3])
        ϕ = two_point_l_upwind(flux, v[i_cell_1], v₂, n)
        # v₂ = SVector(v[i_cell_1][1], 0.0, v[i_cell_1][3])
        # ϕ = flux(v₂, n)
        return ϕ
    end
end

function test_piston()
    model = TwoFluidMixture(IsothermalStiffenedGas(300.0, 1.0, 1.0),
                            IsothermalStiffenedGas(1500.0, 1000.0, 1.0))
    mesh = CartesianMesh(100)

    initial_state(u, ξ) = let p=1.0
        r = ρ(model, p, ξ)
        SVector(r, r*u, r*ξ)
    end
    piston = (0.4, 0.6)
    slope = (0.65, 0.95)
    v₀ = [x < piston[1] ? initial_state(0.1, 1.0) :
          x < piston[2] ? initial_state(0.1, 0.0) :
          x < slope[1]  ? initial_state(0.1, 1.0) :
          x < slope[2]  ? initial_state(0.1*(x-slope[2])/(slope[1]-slope[2]), 1.0) :
                          initial_state(0.0, 1.0)
          for x in cell_centers(mesh)]

    time_step = FixedCourant(0.6)
    nb_time_steps = 50000
    numerical_flux = LUpwind()
    boundary_flux = PistonBC()

    p_wall = Vector{Float64}(undef, nb_time_steps)
    t_wall = Vector{Float64}(undef, nb_time_steps)
    cb(i, t, v) = (vi = v[end]; p_wall[i] = invert_p(model, vi[1], vi[3]/vi[1]); t_wall[i] = t)

    t, v = FiniteVolumes.run(EulerFlux(model), mesh, v₀; time_step, nb_time_steps, numerical_flux, boundary_flux, callback=cb)

    plot(t_wall, p_wall)

    # plot()
    # u_initial = (vi -> vi[2]/vi[1]).(v₀)
    # u_final = (vi -> vi[2]/vi[1]).(v)
    # plot!(mesh, [u_initial u_final], label=["u Initial" "u Final"])

    # ξ_initial = (vi -> vi[3]/vi[1]).(v₀)
    # ξ_final = (vi -> vi[3]/vi[1]).(v)
    # plot!(mesh, [ξ_initial ξ_final], label=["ξ Initial" "ξ Final"])

    # p_final = (vi -> invert_p(model, vi[1], vi[3]/vi[1])).(v)
    # p_final ./= maximum(p_final)
    # plot!(mesh, p_final, label="p final")
end
# test_piston()

##############################################

