
struct LocalState{model, T}
    main::SVector{nb_vars(model), T}
    supp::SVector{nb_varsupp(model), T}
end

struct DataOnMesh{meshtype, modeltype, floattype}
    grid::meshtype
    states::Vector{LocalState{modeltype, floattype}}
end

function local_upwind_flux(w₁::LocalState, w₂::LocalState)
    λ = w₁.supp[1]
    ϕ = flux(model, λ > 0 ? w₁ : w₂)
    return ϕ
end

no_reconstruction(w::DataOnMesh, i_cell, i_face) = w.states[i_cell]

function in_local_coordinates(f, w::DataOnMesh, i_face;
                              reconstruction=no_reconstruction,)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(w.grid, i_face)
    w₁ = rotate_state(reconstruction(w, i_cell_1, i_face), rotation_matrix(w.grid, i_face))
    w₂ = rotate_state(reconstruction(w, i_cell_2, i_face), rotation_matrix(w.grid, i_face))
    ϕ = f(w₁, w₂)
    ϕ = rotate_flux(ϕ, transpose(rotation_matrix(w.grid, i_face)))
    return ϕ
end

function vof_flux(w::DataOnMesh{M, ScalarLinearAdvection, T}, i_face, t, dt) where {M, T}
    α = (x -> x[1]).(w.states[stencil(w.grid)])
    #= ϕ = flux(ScalarLinearAdvectio) =#
    #= rotate_flux(SVector{1, T}(vof_α)) =#
end

function dt_from_cfl(cfl, w:DataOnMesh)
    λmax = maximum(
                   maximum(abs.(in_local_coordinates(compute_w_int |> eigenvalues, grid, i_face)))
                   for i_face in 1:nb_faces(grid)
                  )
                   
    return cfl * smallest_cell(grid)/biggest_face_area(grid)
end

function div(model, grid, numerical_flux, boundary_flux)
    function balance(w::AbstractVector{LocalState}, t=nothing, tdt=nothing)
        Δv = zeros(SVector{nb_vars(model), eltype(model)}, nb_cells(grid))

        for i_face in inner_faces(grid)
            ϕ = numerical_flux(grid, model, w, i_face, t, tdt)
            i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
            Δv[i_cell_1] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_1)
            Δv[i_cell_2] += ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_2)
        end

        for i_face in boundary_faces(grid)
            ϕ = boundary_flux(grid, model, w, wsupp, i_face, t, tdt)
            i_cell = cell_next_to_boundary_face(grid, i_face)
            Δv[i_cell] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell)
        end
        return Δv
    end
end

function using_conservative_variables!(update!, w::AbstractVector{LocalState})
    v = compute_v.(w)
    update!(v)
    set_from_v!.(w, v)
end

function run()
    for i_step in 1:nb_steps
        dt = dt_from_cfl(cfl, grid, w)
        using_conservative_variables!(w) do v
            v += - dt * div(F, grid, FVCF(), NeumannBC())(w, t, tdt)
        end
    end
end
