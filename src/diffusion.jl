import FiniteVolumes.CartesianMeshes.AbstractCartesianMesh

struct DiffusionFlux <: AbstractFlux end
const ∇ = DiffusionFlux()

function (::Centered)(::DiffusionFlux, mesh::AbstractCartesianMesh, w, i_face)
    n = FiniteVolumes.normal_vector(mesh, i_face)
    if mesh isa AbstractCartesianMesh{1}
        face_dir = 1
    elseif mesh isa AbstractCartesianMesh{2}
        face_dir = FiniteVolumes.CartesianMeshes._direction(i_face)
    end
    i_cell_1, i_cell_2 = FiniteVolumes.cells_next_to_inner_face(mesh, i_face)
    return (w[i_cell_2] - w[i_cell_1])*n[face_dir]/FiniteVolumes.CartesianMeshes.dx(mesh)[face_dir]
end

