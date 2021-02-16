import ProgressMeter

struct FixedCourant{T}
    fixed_courant::T
end

run!(flux, args...; kwargs...) = run!([flux], args...; kwargs...)

function run!(fluxes::Vector, mesh, w, t; nb_time_steps, time_step, verbose=true, callback=nothing, kwargs...)

	if verbose
		p = ProgressMeter.Progress(nb_time_steps, dt=0.1)
	end

    for i_time_step in 1:nb_time_steps

        if time_step isa FixedCourant
            dt = minimum(time_step.fixed_courant/courant(1.0, m, mesh, w) for m in models)
        else
            dt = time_step
        end

        for m in models
            # using_conservative_variables!(m, w) do v
            #     v .-= dt * div(m, mesh, w; dt=dt, kwargs...)
            # end
            v(w) = compute_v(m, w)
            v⁻¹(v) = invert_v(m, v)
            Δv = div(m, mesh, w; dt=dt, kwargs...)
            w .= v⁻¹.(v.(w) .- dt.*Δv)
        end

		if verbose
			ProgressMeter.update!(p, i_time_step)
		end

        t += dt

        if callback != nothing
            callback(i_time_step, t, w)
        end
    end

    return t
end

function run(flux, mesh, w₀; kwargs...)
	w = deepcopy(w₀)
    t = run!(flux, mesh, w, 0.0; kwargs...)
	return t, w
end

