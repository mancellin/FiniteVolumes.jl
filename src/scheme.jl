# SCHEME
abstract type Scheme end

# By default, scheme definition can ignore time step
(scheme::Scheme)(flux, mesh, w, i_face, dt) = scheme(flux, mesh, w, i_face)

####################################
struct Centered <: Scheme end
####################################

function (::Centered)(flux::LinearAdvectionFlux, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    return 0.5*(flux(w[i_cell_1], n) + flux(w[i_cell_2], n))
end

####################################
struct Downwind <: Scheme end
####################################

function (::Downwind)(flux::LinearAdvectionFlux, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    λ = eigvals(flux, (w[i_cell_1] + w[i_cell_2])/2, n)
    return λ < zero(λ) ? flux(w[i_cell_1], n) : flux(w[i_cell_2], n)
end

##################################
struct Upwind <: Scheme end
##################################

# Single-wave problems
function (::Upwind)(flux::LinearAdvectionFlux, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    λ = eigvals(flux, (w[i_cell_1] + w[i_cell_2])/2, n)
    return λ > zero(λ) ? flux(w[i_cell_1], n) : flux(w[i_cell_2], n)
end

# Multi-wave problems
function (::Upwind)(flux, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    w_mean = (w[i_cell_1] + w[i_cell_2])/2
    λ, R = eigen(flux, w_mean, n)
    left_to_right = λ .> zero(λ)
    if all(left_to_right)
        return flux(w[i_cell_1], n)
    elseif !any(left_to_right)
        return flux(w[i_cell_2], n)
    else
        f₁ = flux(w[i_cell_1], n)
        f₂ = flux(w[i_cell_2], n)
        decentrement = R \ (f₁ - f₂)
        return (f₁ + f₂ + R*diagm(sign.(λ))*decentrement)/2
        # L = inv(R)
        # return (f₁ + f₂ + R*diagm(sign.(λ))*L*(f₁ - f₂))/2
    end
end

###################################
abstract type BoundaryCondition end
###################################

(scheme::BoundaryCondition)(model, mesh, w, i_face, dt) = scheme(model, mesh, w, i_face)

#########################################
struct NeumannBC <: BoundaryCondition end
#########################################

function (::NeumannBC)(flux, mesh, w, i_face)
    i_cell_1 = cell_next_to_boundary_face(mesh, i_face)
    return flux(w[i_cell_1], normal_vector(mesh, i_face))
end

################################################################################
#                                     Div                                      #
################################################################################

numerical_flux(flux, mesh, w, Φ, i_face, dt) = Φ(flux, mesh, w, i_face, dt)

function div!(Δw, flux, mesh, w, scheme::Scheme, dt=0.0)
    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(flux, mesh, w, scheme, i_face, dt)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δw[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δw[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end
end

function div!(Δw, flux, mesh, w, scheme::BoundaryCondition, dt=0.0)
    @inbounds for i_face in boundary_faces(mesh)
        ϕ = numerical_flux(flux, mesh, w, scheme, i_face, dt)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δw[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
end

function div!(Δw, flux, mesh, w, schemes::NTuple{N, Union{Scheme, BoundaryCondition}}, dt=0.0) where N
    for scheme in schemes
        div!(Δw, flux, mesh, w, scheme, dt)
    end
end

flux_in_cell(flux, mesh, w, scheme, dt, i_face, i_cell) = numerical_flux(flux, mesh, w, scheme, i_face, dt) * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
function Δw_type(flux, mesh, w, schemes::NTuple{N, Union{Scheme, BoundaryCondition}}, dt) where N
    T = Base.return_types(flux_in_cell, typeof.((flux, mesh, w, schemes[1], dt, first(inner_faces(mesh)), first(all_cells(mesh)))))[1]
    if T in (Union{}, Any)
        return eltype(w)
    else
        return T
    end
end
Δw_type(f, m, w, s::Union{Scheme, BoundaryCondition}, dt) = Δw_type(f, m, w, (s,), dt)

function div(flux, mesh, w, schemes::NTuple{N, Union{Scheme, BoundaryCondition}}, dt=0.0) where N
    Δw = zeros(Δw_type(flux, mesh, w, schemes, dt), size(w))
    div!(Δw, flux, mesh, w, schemes, dt)
    return Δw
end

div(flux, mesh, w, scheme::Union{Scheme, BoundaryCondition}, dt=0.0) = div(flux, mesh, w, (scheme,), dt)

