
function courant(Δt, flux, mesh, w)
    courant = 0.0
    for i_face in inner_faces(mesh)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        w₁ = rotate_state(w[i_cell_1], model, rotation_matrix(mesh, i_face))
        w₂ = rotate_state(w[i_cell_2], model, rotation_matrix(mesh, i_face))
        local_model = rotate_model(model, rotation_matrix(mesh, i_face), face_center(mesh, i_face))
        w_int = compute_w_int(local_model, w₁, w₂)
        λ = eigenvalues(local_model, w_int)
        maxλ = maximum(abs.(λ))
        courant = max(courant, maxλ * Δt * face_area(mesh, i_face) / max(cell_volume(mesh, i_cell_1), cell_volume(mesh, i_cell_2)))
    end
    return courant
end

function courant(Δt, flux::LinearAdvectionFlux, mesh::AbstractCartesianMesh{1}, w)
    return abs(flux.velocity[1]) * Δt / dx(mesh)[1]
end

# function courant(Δt, flux::LinearAdvectionFlux, mesh::PeriodicRegularMesh2D, w)
#     vx, vy = model.velocity
#     return max(abs(vx) * Δt / dx(mesh), abs(vy) * Δt / dy(mesh))
# end
