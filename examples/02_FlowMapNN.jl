# Activate environment and install dependencies defined in Project.toml
include("../env/activate_env.jl")

using Random
using Distributions
using Lux
using CUDA
using Zygote
using Optimisers
using Random, Statistics, Printf
using CairoMakie
using DifferentialEquations

# ===# Define computation devices #===
# Use CPU for prediction and evaluation
const cdev = cpu_device()

# Use a Reactant-compatible device (e.g., GPU if available) for training
function get_device()
    if CUDA.has_cuda()
        Reactant.set_default_backend("gpu") 
        return reactant_device()
    else 
        return cdev
    end
end

const dev_training = get_device()

function lorenz(du, u, p, t)
    σ = 10.0
    r = 28.0
    b = 8/3

    x, y, z = u

    du[1] = dxdt = σ*(y - x)
    du[2] = dydt = r*x - y - x*z
    du[3] = dzdt = x*y - b*z
end

dt = 0.01
T = 8.0
tspan = (0.0, T)
t = tspan[1]:dt:tspan[2]
n_trajectories = 100

input_data = zeros(Float32, (3, length(t)-1, n_trajectories))
output_data = zeros(Float32, (3, length(t)-1, n_trajectories))


fig = Figure()
ax = Axis3(fig[1, 1], title="Lorenz Attractor", xlabel="X", ylabel="Y", zlabel="Z")

for j in 1:n_trajectories
    u₀ = -15 .+ 30 .* rand(3)
    prob = ODEProblem(lorenz, u₀, tspan)
    sol = solve(prob, Tsit5(); saveat=t, abstol=1e-11, reltol=1e-10)

    input_data[:, :, j] .= sol[:, 1:end-1]
    output_data[:, :, j] .= sol[:, 2:end]

    lines!(ax, sol[1, :], sol[2, :], sol[3, :], linewidth=1.5, transparency=true, alpha=0.7)
    scatter!(ax, [sol[1, 1]], [sol[2, 1]], [sol[3, 3]], markersize=8)
end
fig

input_data = reshape(input_data, (:, n_trajectories))
output_data = reshape(output_data, (:, n_trajectories))

# ---
# FNN

model = Chain(
    Dense(2400, 2400*3, sigmoid),
    Dense(2400*3, 2400*3, relu),
    Dense(2400*3, 2400)
)

rng = Random.default_rng()
Random.seed!(7)

function train(model, device, rng, kwargs...)
    lr = 1e-3
    n_epochs = 1000
    batch_size = 32
    optimizer = Adam(lr)
    ps, st = Lux.setup(rng, model) |> device
    vjp = AutoZygote()
    lossfn = MSELoss()

    train_state = Training.TrainState(model, ps, st, optimizer)

    tr_acc = 0.0
    stime = time()
    for epoch in 1:n_epochs
        loss_acc = 0.0
        perm = randperm(n_trajectories)
        for batch in Iterators.partition(perm, batch_size)
            x_batch = input_data[:, batch] |> device
            y_batch = output_data[:, batch] |> device
    
            _, loss, _, train_state = Training.single_train_step!(
                vjp, lossfn, (x_batch, y_batch), train_state
            )
            loss_acc += loss
        end
        if epoch % 100 == 0
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss_acc
        end
    end
    ttime = time() - stime
    return train_state, tr_acc, ttime
end

tstate, tr_acc, tr_time = train(model, dev_training, rng)

# ---
# Test
T = 8.0
tspan = (0.0, T)
t = tspan[1]:dt:tspan[2]
n_trajectories = 10

input_data = zeros(Float32, (3, length(t)-1, n_trajectories))
nn_sol = zeros(Float32, (3, length(t)-1, n_trajectories))

fig = Figure(resolution=(800, 600))
ax = Axis3(fig[1, 1], title="Lorenz Attractor", xlabel="X", ylabel="Y", zlabel="Z")

for j in 1:n_trajectories
    u₀ = -15 .+ 30 .* rand(3)
    prob = ODEProblem(lorenz, u₀, tspan)
    sol = solve(prob, Tsit5(); saveat=t, abstol=1e-11, reltol=1e-10)

    input_data[:, :, j] .= sol[:, 1:end-1]

    # Neural Network prediction
    nn_input = reshape(sol[:, 1:end-1], :, 1) |> dev_training
    y_pred, _ = Lux.apply(
        tstate.model, nn_input, tstate.parameters, Lux.testmode(tstate.states)
    )
    nn_sol[:, :, j] .= reshape(Array(y_pred), (3, 800))

    # Plot ground truth (Runge-Kutta)
    lines!(ax, input_data[1, :, j], input_data[2, :, j], input_data[3, :, j],
        color=:blue, linewidth=1.5, alpha=0.7)

    # Plot neural network prediction
    lines!(ax, nn_sol[1, :, j], nn_sol[2, :, j], nn_sol[3, :, j],
        color=:red, linewidth=1.5, alpha=0.7)
end
fig


# Display for Each X, Y, Z variables
fig = Figure(resolution=(1000, 800))

# X subplot
axX = Axis(fig[1, 1], title="Lorenz Attractor - X", xlabel="t", ylabel="X")
lines!(axX, t[1:end-1], input_data[1, :, 1], color=:blue, linewidth=1.5, alpha=0.7, label="Lorenz")
lines!(axX, t[1:end-1], nn_sol[1, :, 1], color=:red, linewidth=1.5, alpha=0.7, label="NN")
axislegend(axX)

# Y subplot
axY = Axis(fig[2, 1], title="Lorenz Attractor - Y", xlabel="t", ylabel="Y")
lines!(axY, t[1:end-1], input_data[2, :, 1], color=:blue, linewidth=1.5, alpha=0.7, label="Lorenz")
lines!(axY, t[1:end-1], nn_sol[2, :, 1], color=:red, linewidth=1.5, alpha=0.7, label="NN")
axislegend(axY)

# Z subplot
axZ = Axis(fig[3, 1], title="Lorenz Attractor - Z", xlabel="t", ylabel="Z")
lines!(axZ, t[1:end-1], input_data[3, :, 1], color=:blue, linewidth=1.5, alpha=0.7, label="Lorenz")
lines!(axZ, t[1:end-1], nn_sol[3, :, 1], color=:red, linewidth=1.5, alpha=0.7, label="NN")
axislegend(axZ)

fig