
struct UnidirectionalFlux{D, F}
    flux::F
end
UnidirectionalFlux{D}(f) where D = UnidirectionalFlux{D, eltype(f)}(f)

function directional_splitting(f::LinearAdvectionFlux{T}) where T<:SVector{2}
    z = zero(eltype(f.velocity))
    (UnidirectionalFlux{1}(LinearAdvectionFlux(SVector(f.velocity[1], z))),
     UnidirectionalFlux{2}(LinearAdvectionFlux(SVector(z, f.velocity[2]))))
end

function directional_splitting(f::LinearAdvectionFlux{T}) where T<:SVector{3}
    z = zero(eltype(f.velocity))
    (UnidirectionalFlux{1}(LinearAdvectionFlux(SVector(f.velocity[1], z, z))),
     UnidirectionalFlux{2}(LinearAdvectionFlux(SVector(z, f.velocity[2], z))),
     UnidirectionalFlux{3}(LinearAdvectionFlux(SVector(z, z, f.velocity[3]))))
end

function directional_splitting(f::AdvectionFlux{2})
    (
    UnidirectionalFlux{1}(AdvectionFlux((m, i_face) -> SVector(flux.velocity(m, i_face)[1], 0.0))),
    UnidirectionalFlux{2}(AdvectionFlux((m, i_face) -> SVector(0.0, flux.velocity(m, i_face)[2])))
    )
end


function numerical_flux(wrapped::UnidirectionalFlux{D}, mesh::CartesianMeshes.AbstractCartesianMesh, w, scheme, i_face, dt) where D
    if CartesianMeshes._direction(i_face) == D
        numerical_flux(wrapped.flux, mesh, w, scheme, i_face, dt)
    else
        zero(eltype(w))
    end
end


# ANOTHER OPTION: OVERLOADING DIV TO LOOP ONLY ON SOME FACES
# function faces_in_direction(::Val{D}, mesh::CartesianMeshes.AbstractCartesianMesh) where D
#     ...
# end
# function div!(Δw, wrapped::UnidirectionalFlux{D}, mesh::CartesianMeshes.AbstractCartesianMesh, w, scheme, i_face, dt) where D
#     @inbounds for i_face in faces_in_direction(Val(D), mesh)
#         ϕ = numerical_flux(flux, mesh, w, scheme, i_face, dt)
#         i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
#         Δw[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
#         Δw[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
#     end
# end


function courant(Δt, wrapped::UnidirectionalFlux{D}, mesh, w, i_face) where D
    if CartesianMeshes._direction(i_face) == D
        courant(Δt, wrapped.flux, mesh, w, i_face)
    else
        zero(Δt)
    end
end
