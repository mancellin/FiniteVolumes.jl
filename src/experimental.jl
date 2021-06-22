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
