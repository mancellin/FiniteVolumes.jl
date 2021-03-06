# SCHEME
abstract type NumericalFlux end

# By default, can ignore time step
(scheme::NumericalFlux)(flux, mesh, w, i_face, dt) = scheme(flux, mesh, w, i_face)

####################################
struct Centered <: NumericalFlux end
####################################

function (::Centered)(flux::Union{LinearAdvectionFlux, FluxFunction{<:Number}}, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    return 0.5*(flux(w[i_cell_1], n) + flux(w[i_cell_2], n))
end

####################################
struct Downwind <: NumericalFlux end
####################################

function (::Downwind)(flux::Union{LinearAdvectionFlux, FluxFunction{<:Number}}, mesh, w, i_face)
    n = normal_vector(mesh, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    λ = eigvals(flux, (w[i_cell_1] + w[i_cell_2])/2, n)
    return λ < zero(λ) ? flux(w[i_cell_1], n) : flux(w[i_cell_2], n)
end

##################################
struct Upwind <: NumericalFlux end
##################################

# Single-wave problems
function (::Upwind)(flux::Union{LinearAdvectionFlux, FluxFunction{<:Number}}, mesh, w, i_face)
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

div!(Δw, flux, mesh, w, args...) = div!(Δw, FluxFunction{eltype(w), nb_dims(mesh), typeof(flux)}(flux), mesh, w, args...)

function div!(Δw, flux::AbstractFlux, mesh, w, numerical_flux::NumericalFlux, dt=0.0)
    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(flux, mesh, w, i_face, dt)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δw[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δw[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end
end

function div!(Δw, flux::AbstractFlux, mesh, w, boundary_flux::BoundaryCondition, dt=0.0)
    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(flux, mesh, w, i_face, dt)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δw[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
end

flux_in_cell(flux, mesh, w, scheme, dt, i_face, i_cell) = scheme(flux, mesh, w, i_face, dt) * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
function Δw_type(flux, mesh, w, scheme, dt)
    T = Base.return_types(flux_in_cell, typeof.((flux, mesh, w, scheme, dt, first(inner_faces(mesh)), first(all_cells(mesh)))))[1]
    if T in (Union{}, Any)
        return eltype(w)
    else
        return T
    end
end

function div(flux, mesh, w, numerical_flux::Union{NumericalFlux, BoundaryCondition}, dt=0.0)
    Δw = zeros(Δw_type(flux, mesh, w, numerical_flux, dt), size(w))
    div!(Δw, flux, mesh, w, numerical_flux, dt)
    return Δw
end

function div(flux, mesh, w, dt=0.0; numerical_flux=Upwind(), boundary_flux=NeumannBC())
    Δw = zeros(Δw_type(flux, mesh, w, numerical_flux, dt), size(w))
    div!(Δw, flux, mesh, w, numerical_flux, dt)
    div!(Δw, flux, mesh, w, boundary_flux, dt)
    return Δw
end

