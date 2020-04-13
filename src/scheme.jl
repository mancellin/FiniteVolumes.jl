using ProgressMeter: @showprogress


# NUMERICAL FLUX
function local_upwind_flux(model::ScalarLinearAdvection, w₁, wsupp₁, w₂, wsupp₂)
    λ = model.velocity[1]
    ϕ = normal_flux(model, λ > 0 ? w₁ : w₂, wsupp₁)
    return ϕ, abs(λ)
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

    return ϕ, maximum(abs.(λ))
end

function no_reconstruction(grid, model, w, wsupp, i_cell, i_face)
    return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
end

function in_local_coordinates(f, grid, model, w, wsupp, i_face; reconstruction=no_reconstruction)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell_1, i_face)
    w₂, wsupp₂ = reconstruction(grid, model, w, wsupp, i_cell_2, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ, newλmax = f(local_model, w₁, wsupp₁, w₂, wsupp₂)
    ϕ = rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
    return ϕ, newλmax
end

first_order_upwind(args...) = in_local_coordinates(local_upwind_flux, args...; reconstruction=no_reconstruction)

# BOUNDARY FLUX
function in_local_coordinates_at_boundary(f, grid, model, w, wsupp, i_face; reconstruction=no_reconstruction)
    i_cell = cell_next_to_boundary_face(grid, i_face)
    w₁, wsupp₁ = reconstruction(grid, model, w, wsupp, i_cell, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    ϕ, newλmax = f(local_model, w₁, wsupp₁)
    ϕ = rotate_flux(ϕ, model, transpose(rotation_matrix(grid, i_face)))
    return ϕ, newλmax
end

neumann_bc(args...) = in_local_coordinates_at_boundary(args...; reconstruction=no_reconstruction) do model, w₁, wsupp₁
    (normal_flux(model, w₁, wsupp₁), maximum(abs.(eigenvalues(model, w₁, wsupp₁))))
end

# BALANCE
function balance(model, grid, w, wsupp;
                 numerical_flux=first_order_upwind,
                 boundary_flux=neumann_bc,
                )
    Δv = zeros(eltype(w), nb_cells(grid))

    λmax = 0.0
    @inbounds for i_face in inner_faces(grid)
        ϕ, newλmax = numerical_flux(grid, model, w, wsupp, i_face)
        i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
        Δv[i_cell_1] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_1)
        Δv[i_cell_2] += ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell_2)
        λmax = max(λmax, newλmax)
    end

    @inbounds for i_face in boundary_faces(grid)
        ϕ, newλmax = boundary_flux(grid, model, w, wsupp, i_face)
        i_cell = cell_next_to_boundary_face(grid, i_face)
        Δv[i_cell] -= ϕ * face_area(grid, i_face) / cell_volume(grid, i_cell)
        λmax = max(λmax, newλmax)
    end

    return (Δv, λmax)
end

div(model, mesh; numerical_flux=first_order_upwind) = w -> begin
    wsupp = map(wi -> FiniteVolumes.compute_wsupp(model, wi), w)
    return -balance(model, mesh, w, wsupp, numerical_flux=numerical_flux)[1]
end

dt_from_cfl(cfl, grid, λmax)::Float64 = cfl * minimum((cell_volume(grid, i_cell) for i_cell in 1:nb_cells(grid))) / (maximum(face_area(grid, i_face) for i_face in 1:nb_faces(grid)) * λmax)

cfl_from_dt(dt, grid, λmax)::Float64 = dt * (maximum(face_area(grid, i_face) for i_face in 1:nb_faces(grid)) * λmax) / minimum((cell_volume(grid, i_cell) for i_cell in 1:nb_cells(grid))) 

isnothing(x::Nothing) = true
isnothing(x::Any) = false

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

function update!(model, grids::Union{Tuple, Vector, Set}, w, wsupp; dt=nothing, cfl=nothing, kwargs...)
    # Set time step on first direction
	(dt, cfl) = update!(model, grids[1], w, wsupp; dt=dt, cfl=cfl, kwargs...)
    for grid in grids[2:end]
		(dt_i, cfl_i) = update!(model, grid, w, wsupp; dt=dt, kwargs...)
        cfl = max(cfl_i, cfl)
    end
    return (dt, cfl)
end

function run!(model, grid, w, wsupp; nb_time_steps, kwargs...)
    t = 0.0
    @showprogress 0.1 "Running " for i_time_step in 1:nb_time_steps
        (dt, cfl) = update!(model, grid, w, wsupp; kwargs...)
		if i_time_step == 1
			println("CFL: $cfl")
		end
		if cfl > 1.0
			println("!!! CFL: $cfl !!!")
		end
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

