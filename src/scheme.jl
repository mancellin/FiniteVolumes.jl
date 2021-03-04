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

# function in_local_coordinates(f, model, mesh, w, i_face)
#     i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
#     w₁ = rotate_state(w[i_cell_1], model, rotation_matrix(mesh, i_face))
#     w₂ = rotate_state(w[i_cell_2], model, rotation_matrix(mesh, i_face))
#     local_model = rotate_model(model, rotation_matrix(mesh, i_face), face_center(mesh, i_face))
#     ϕ = f(local_model, w₁, w₂)
#     return rotate_flux(ϕ, model, transpose(rotation_matrix(mesh, i_face)))
# end

# function (::Upwind)(model, mesh, w, i_face)  # l-upwind FVCF
#     in_local_coordinates(model, mesh, w, i_face) do local_model, w₁, w₂
#         flux₁ = normal_flux(local_model, w₁)
#         flux₂ = normal_flux(local_model, w₂)

#         w_int = compute_w_int(local_model, w₁, w₂)
#         λ = eigenvalues(local_model, w_int)

#         L₁ = left_eigenvectors(local_model, w₁)
#         L₂ = left_eigenvectors(local_model, w₂)

#         L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
#         L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

#         ϕ = L_upwind \ L_flux_upwind
#         return ϕ
#     end
# end

# BC

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

function div!(Δw, flux, mesh, w, numerical_flux::NumericalFlux, dt=0.0)
    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(flux, mesh, w, i_face, dt)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δw[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δw[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end
end

function div!(Δw, flux, mesh, w, boundary_flux::BoundaryCondition, dt=0.0)
    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(flux, mesh, w, i_face, dt)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δw[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
end

flux_in_cell(flux, mesh, w, scheme, dt, i_face, i_cell) = scheme(flux, mesh, w, i_face, dt) * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
Δw_type(flux, mesh, w, scheme, dt) = Base.return_types(flux_in_cell, typeof.((flux, mesh, w, scheme, dt, first(inner_faces(mesh)), first(all_cells(mesh)))))[1]

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

function numerical_fluxes!(Φ, flux, mesh, w, numerical_flux::NumericalFlux, dt=0.0)
    map!(i_face -> numerical_flux(flux, mesh, w, i_face, dt), Φ, collect(inner_faces(mesh)))
end

function numerical_fluxes(flux, mesh, w, numerical_flux::NumericalFlux, dt=0.0)
    map(i_face -> numerical_flux(flux, mesh, w, i_face, dt), collect(inner_faces(mesh)))
end

"""Sparse matrix to sum the flux on both side of a face and get the change in a cell.

In 1D: sparse matrix.
For 2D cartesian mesh, it should be a higher order mesh, but since there is no higher dimensional sparse array, the field needs to be reshaped to a vector...
"""
function inner_faces_to_cells_matrix(mesh::AbstractCartesianMesh)
    D = spzeros(FiniteVolumes.nb_cells(mesh), FiniteVolumes.nb_inner_faces(mesh))
    li = LinearIndices(cell_centers(mesh))
    for (i, face) in enumerate(FiniteVolumes.inner_faces(mesh))
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, face)
        D[li[i_cell_1], i] -= face_area(mesh, face) / cell_volume(mesh, i_cell_1)
        D[li[i_cell_2], i] += face_area(mesh, face) / cell_volume(mesh, i_cell_1)
    end
    D
end

function update!(w, Φ, mesh, dt=0.0)
    for (i, face) in enumerate(inner_faces(mesh))
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, face)
        w[i_cell_1] -= dt*Φ[i] * face_area(mesh, face) / cell_volume(mesh, i_cell_1)
        w[i_cell_2] += dt*Φ[i] * face_area(mesh, face) / cell_volume(mesh, i_cell_2)
    end
end

