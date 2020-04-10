abstract type AbstractMesh{Dim} end
abstract type AbstractModel{NbVars, Dim} end

struct FVCF end

function numerical_normal_flux(model::AbstractModel, method::FVCF, mesh, w, i_face)
    in_local_coordinates(mesh, model, w, i_face) do local_model, w₁, w₂
        flux₁ = x_flux(local_model, w₁)
        flux₂ = x_flux(local_model, w₂)

        w_int = compute_w_int(local_model, w₁, w)
        λ = eigenvalues(local_model, w_int)

        L₁ = left_eigenvectors(local_model, w₁)
        L₂ = left_eigenvectors(local_model, w₂)

        L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
        L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

        ϕ = L_upwind \ L_flux_upwind
        return ϕ
    end
end


struct NeumannBC end

function boundary_normal_flux(model::AbstractModel, bc::NeumannBC, mesh, w, i_face)
    in_local_coordinates_at_boundary(mesh, model, w, i_face) do local_model, w₁
        flux(local_model, w₁, face_center(mesh, i_face), normal(mesh, i_face))
    end
end


"""
    div(F::AbstractModel, mesh; method=FVCF(), bc=NeumannBC())
    where F(w::SVector{N, T}, [x::SVector{D, T}])::SMatrix{(N, D), T}
"""
function div(model::AbstractModel{N, D}, mesh; method=FVCF(), bc=NeumannBC()) where {N, D}
    function divF(w)
        Δv = zeros(SVector{N, eltype(w)}, nb_cells(mesh))

        @inbounds for i_face in inner_faces(mesh)
            ϕ = numerical_normal_flux(Fn, method, mesh, w, i_face)
            i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
            Δv[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
            Δv[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
        end

        @inbounds for i_face in boundary_faces(mesh)
            ϕ = boundary_normal_flux(Fn, bc, mesh, w, i_face)
            i_cell = cell_next_to_boundary_face(mesh, i_face)
            Δv[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
        end
        return Δv 
    end
end

struct AnonymousScalarModel{N, D}
    flux::Function
end

AnonymousScalarModel(f::Function, dim) = AnonymousModel(f , signature(f), dim)

function AnonymousScalarModel(f::Function, sig::Signature{1, 1}, dim::Int)

end

div(F::Function, mesh::AbstractMesh{Dim}; kwargs...) where Dim = div(AnonymousModel(F, Dim), mesh; kwargs...)






function using_conservative_variables!(update!, model::AbstractModel, w::AbstractVector{LocalState})
    v = compute_v.(model, w)
    update!(v)
    set_from_v!.(w, model, v)
end

function using_conservative_variables!(update!, v_from_w::Function, w::AbstractVector{LocalState})
    v = v_from_w.(w)
    update!(v)
    for i_cell in 1:length(w)
        w[i_cell] = Roots.find_zero(w -> v_from_w(w) - v[i_cell], w[i_cell], Roots.Newton())
    end
end
