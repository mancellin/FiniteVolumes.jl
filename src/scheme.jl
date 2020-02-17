using ProgressMeter: @showprogress


function numerical_flux(model::ScalarLinearAdvection, w₁, wsupp₁, w₂, wsupp₂)
    λ = wsupp₁[1]
    ϕ = flux(model, λ > 0 ? w₁ : w₂, wsupp₁)
    return ϕ, abs(λ)
end


function numerical_flux(model, w₁, wsupp₁, w₂, wsupp₂)
    flux₁ = flux(model, w₁, wsupp₁)
    flux₂ = flux(model, w₂, wsupp₂)

    w_int, wsupp_int = compute_w_int(model, w₁, wsupp₁, w₂, wsupp₂)
    λ = eigenvalues(model, w_int, wsupp_int)

    L₁ = left_eigenvectors(model, w₁, wsupp₁)
    L₂ = left_eigenvectors(model, w₂, wsupp₂)

    L_upwind = ifelse.(λ .> 0.0, L₁, L₂)
    L_flux_upwind = ifelse.(λ .> 0.0, L₁ * flux₁, L₂ * flux₂)

    ϕ = L_upwind \ L_flux_upwind

    return ϕ, maximum(abs.(λ))
end

# Neuman boundary flux
bc_flux(model, w, wsupp) = (flux(model, w, wsupp), maximum(abs.(eigenvalues(model, w, wsupp))))


function no_reconstruction(grid, model, w, wsupp, i_cell, i_face)
    return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
end

function balance(model, grid, w, wsupp; reconstruction=no_reconstruction)
    Δv = Vector{SVector{nb_vars(model), eltype(model)}}(undef, nb_cells(grid))
    @inbounds for i_cell in 1:nb_cells(grid)
        Δv[i_cell] = zeros(SVector{nb_vars(model), eltype(model)})
    end

    λmax = 0.0
    @inbounds for i_face in inner_faces(grid)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
        w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell_1, i_face)
        w₂, wsupp₂ = reconstruction(grid, model, w, wsupp, i_cell_2, i_face)
        ϕ, newλmax = numerical_flux(model, w₁, wsupp₁, w₂, wsupp₂)
        ϕ = rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
        Δv[i_cell_1] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_1)
        Δv[i_cell_2] += ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_2)
        λmax = max(λmax, newλmax)
    end

    @inbounds for i_face in boundary_faces(grid)
        i_cell = cell_next_to_boundary_face(grid, i_face)
        w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell, i_face)
        ϕ, newλmax = bc_flux(model, w₁, wsupp₁)
        ϕ = rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
        Δv[i_cell] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell)
        λmax = max(λmax, newλmax)
    end

    return (Δv, λmax)
end

dt_from_cfl(cfl, grid, λmax)::Float64 = cfl * minimum((cell_volume(grid, i_cell) for i_cell in 1:nb_cells(grid))) / (maximum(face_area(grid, i_face) for i_face in 1:nb_faces(grid)) * λmax)

cfl_from_dt(dt, grid, λmax)::Float64 = dt * (maximum(face_area(grid, i_face) for i_face in 1:nb_faces(grid)) * λmax) / minimum((cell_volume(grid, i_cell) for i_cell in 1:nb_cells(grid))) 

isnothing(x::Nothing) = true
isnothing(x::Any) = false

function dt_and_cfl(dt, cfl, grid, λmax)
	if isnothing(dt)
		if isnothing(cfl)
			error("You need to specify either a time step or a CFL")
		else
			return (dt_from_cfl(cfl, grid, λmax), cfl)
		end
	else
		if isnothing(cfl)
			return (dt, cfl_from_dt(dt, grid, λmax))
		else
			error("You need to specify a time step or a CFL, but not both")
		end
	end
end

function update!(model, grid, w, wsupp; dt=nothing, cfl=nothing, kwargs...)
    Δv, λmax = balance(model, grid, w, wsupp; kwargs...)
	(dt, cfl) = dt_and_cfl(dt, cfl, grid, λmax)
    @inbounds for i_cell in 1:nb_cells(grid)
        new_v =  compute_v(model, w[i_cell], wsupp[i_cell]) + dt * Δv[i_cell]
        w[i_cell] = invert_v(model, new_v)
        wsupp[i_cell] = compute_wsupp(model, w[i_cell])
    end
    return (dt, cfl)
end

function update!(model, grids::Union{Tuple, Vector, Set}, w, wsupp; kwargs...)
    # Set time step on first direction
	(dt_1, cfl) = update!(model, grids[1], w, wsupp; kwargs...)
    for grid in grids[2:end]
		(dt_i, cfl_i) = update!(model, grid, w, wsupp; dt=dt_1, reconstruction=reconstruction)
        cfl = max(cfl_i, cfl)
    end
    return (dt, cfl)
end

function run!(model, grid, w, wsupp; nb_time_steps, kwargs...)
    t = 0.0
    @showprogress 0.1 "Running " for i_time_step in 1:nb_time_steps
        (dt, cfl) = update!(model, grid, w, wsupp; kwargs...)
        t += dt
    end
    return t
end

