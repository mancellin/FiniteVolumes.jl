function courant(Δt, flux, mesh, w)
    c = zero(eltype(Δt))
    for i_face in inner_faces(mesh)
        c = max(c, courant(Δt, flux, mesh, w, i_face))
    end
    return c
end

function courant(Δt, flux, mesh, w, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
    n = normal_vector(mesh, i_face)
    w_mean = (w[i_cell_1] + w[i_cell_2])/2
    λ = eigvals(flux, w_mean, n)
    maxλ = maximum(abs.(λ))
    return maxλ * Δt * face_area(mesh, i_face) / min(cell_volume(mesh, i_cell_1), cell_volume(mesh, i_cell_2))
end

function courant(Δt, flux::AdvectionFlux, mesh, w, i_face)
    v = LinearAdvectionFlux(flux.velocity(mesh, i_face))
    return FiniteVolumes.courant(Δt, v, mesh, w, i_face)
end

# function courant(Δt, flux::LinearAdvectionFlux, mesh::AbstractCartesianMesh{1}, w)
#     return abs(flux.velocity[1]) * Δt / FiniteVolumes.CartesianMeshes.dx(mesh)[1]
# end

# function courant(Δt, flux::LinearAdvectionFlux, mesh::PeriodicRegularMesh2D, w)
#     vx, vy = model.velocity
#     return max(abs(vx) * Δt / dx(mesh), abs(vy) * Δt / dy(mesh))
# end
