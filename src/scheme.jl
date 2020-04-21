using ProgressMeter: @showprogress


# NUMERICAL FLUX
function local_upwind_flux(model::ScalarLinearAdvection, w₁, wsupp₁, w₂, wsupp₂)
    λ = model.velocity[1]
    return normal_flux(model, λ > 0 ? w₁ : w₂, wsupp₁)
end

function local_upwind_flux(model, w₁, wsupp₁, w₂, wsupp₂)  # l-upwind FVCF
    flux₁ = normal_flux(model, w₁, wsupp₁)
    flux₂ = normal_flux(model, w₂, wsupp₂)

    w_int, wsupp_int = compute_w_int(model, w₁, wsupp₁, w₂, wsupp₂)
    λ = eigenvalues(model, w_int, wsupp_int)

    L₁ = left_eigenvectors(model, w₁, wsupp₁)
    L₂ = left_eigenvectors(model, w₂, wsupp₂)

    L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
    L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

    ϕ = L_upwind \ L_flux_upwind

    return ϕ
end

function no_reconstruction(grid, model, w, wsupp, i_cell, i_face)
    return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
end

function in_local_coordinates(f, grid, model, w, wsupp, i_face; reconstruction=no_reconstruction)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell_1, i_face)
    w₂, wsupp₂ = reconstruction(grid, model, w, wsupp, i_cell_2, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ = f(local_model, w₁, wsupp₁, w₂, wsupp₂)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
end

first_order_upwind(args...) = in_local_coordinates(local_upwind_flux, args...; reconstruction=no_reconstruction)


# BOUNDARY FLUX
function in_local_coordinates_at_boundary(f, grid, model, w, wsupp, i_face; reconstruction=no_reconstruction)
    i_cell = cell_next_to_boundary_face(grid, i_face)
    w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ = f(local_model, w₁, wsupp₁)
    return rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
end

neumann_bc(args...) = in_local_coordinates_at_boundary(args...; reconstruction=no_reconstruction) do model, w₁, wsupp₁
    normal_flux(model, w₁, wsupp₁)
end


# BALANCE
function div(model, mesh, w, wsupp; numerical_flux=first_order_upwind, boundary_flux=neumann_bc)
    Δv = zeros(eltype(w), nb_cells(mesh))

    @inbounds for i_face in inner_faces(mesh)
        ϕ = numerical_flux(mesh, model, w, wsupp, i_face)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        Δv[i_cell_1] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_1)
        Δv[i_cell_2] -= ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell_2)
    end

    @inbounds for i_face in boundary_faces(mesh)
        ϕ = boundary_flux(mesh, model, w, wsupp, i_face)
        i_cell = cell_next_to_boundary_face(mesh, i_face)
        Δv[i_cell] += ϕ * face_area(mesh, i_face) / cell_volume(mesh, i_cell)
    end
    return Δv
end

function update!(model, grid, w, wsupp, Δt; kwargs...)
    Δv = div(model, grid, w, wsupp; kwargs...)
    @inbounds for i_cell in 1:nb_cells(grid)
        new_v =  compute_v(model, w[i_cell], wsupp[i_cell]) - Δt * Δv[i_cell]
        w[i_cell] = invert_v(model, new_v)
        wsupp[i_cell] = compute_wsupp(model, w[i_cell])
    end
    #= if i_time_step == 1 =#
    #=     println("CFL: $(courant(Δt, grid, model, w, wsupp))") =#
    #= end =#
    return w, wsupp
end

function update!(model, grids::Union{Tuple, Vector, Set}, w, wsupp, Δt; kwargs...)
    # Set time step on first direction
    for grid in grids
        w, wsupp = update!(model, grid, w, wsupp, Δt; kwargs...)
    end
    return w, wsupp
end


# COURANT

function courant(Δt, mesh, model, w, wsupp)
    courant = 0.0
    for i_face in inner_faces(mesh)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(mesh, i_face)
        w₁, wsupp₁ = rotate_state(w[i_cell_1], wsupp[i_cell_1], model, rotation_matrix(mesh, i_face))
        w₂, wsupp₂ = rotate_state(w[i_cell_2], wsupp[i_cell_2], model, rotation_matrix(mesh, i_face))
        local_model = rotate_model(model, rotation_matrix(mesh, i_face))
        w_int, wsupp_int = compute_w_int(local_model, w₁, wsupp₁, w₂, wsupp₂)
        λ = eigenvalues(local_model, w_int, wsupp_int)
        maxλ = maximum(abs.(λ))
        courant = max(courant, maxλ * Δt * face_area(mesh, i_face) / max(cell_volume(mesh, i_cell_1), cell_volume(mesh, i_cell_2)))
    end
    return courant
end

function run!(model, grid, w, wsupp; nb_time_steps, dt=nothing, cfl=nothing, kwargs...)
    t = 0.0
    @showprogress 0.1 "Running " for i_time_step in 1:nb_time_steps
        if isnothing(dt)
            if !isnothing(cfl)
                dt = cfl/courant(1.0, grid, model, w, wsupp)
            else
                error("No time step nor Courant number has been provided :(")
            end
        end
        w, wsupp = update!(model, grid, w, wsupp, dt; kwargs...)
        t += dt
    end
    return t
end

function run(model, grid, w₀; kwargs...)
	w = deepcopy(w₀)
	wsupp = map(wi -> compute_wsupp(model, wi), w)
	t = run!(model, grid, w, wsupp; kwargs...)
	return t, w
end

