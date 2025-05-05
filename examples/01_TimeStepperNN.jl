# Activate environment and install dependencies defined in Project.toml
include("../env/activate_env.jl")

# Load standard and third-party libraries
using Random                      # For random number generation
using Distributions               # To use probability distributions like Normal()
using Lux                         # Lightweight neural network library
using LuxCUDA
using Zygote                      # AD backend for gradients (used with Lux)
using Optimisers                  # For optimization algorithms like Adam
using Random, Statistics, Printf  # Additional useful stdlibs
using CairoMakie                  # High-quality plotting backend
using Functors                    # For parameter manipulation in models
using Enzyme                      # For compiling and differentiating Julia code
using Reactant                    # For running Lux models on GPU-like accelerators

# Define computation devices
Reactant.set_default_backend("gpu") 
const cdev = cpu_device()         # Use CPU for prediction and evaluation
const xdev = reactant_device()    # Use a Reactant-compatible device (e.g., GPU) for training

# Set up the time step and time range
dt = 0.01
x = 0:dt:50                       # Full time series from t = 0 to 50 with spacing dt
xt = x[2001:4000]                 # Training time range (a subwindow)

# Count number of training and testing samples
n_data_test = length(x)
n_data_train = length(xt)

xt = reshape(xt, (1, n_data_train))  # Make it a 2D array (1 row, n columns)

# Define ground-truth signal: a noisy cosine function with linear trend
a = 11.0
b = π/4
c = 1.5
d = 22.0
y = @. a * cos(b*x) + c * x + d   # Elementwise expression with @. macro

# Add Gaussian noise to the signal
d = Normal()                      # Standard normal distribution
n = length(y)
noise = rand(d, n)               # Generate random noise
ynoise = @. y + 2*noise          # Add scaled noise to signal

# Extract training labels and reshape
yt = ynoise[2001:4000]
yt = reshape(yt, (1, n_data_train))

# Prepare test data (entire range)
x_test = reshape(x, (1, n_data_test))
y_test = reshape(ynoise, (1, n_data_test))

# Define a simple fully-connected feedforward neural network (FNN)
function FNN(input_size, hidden_size, num_layers, output_size)
    model = Chain(
        Dense(input_size, hidden_size, elu),                                # Input layer
        [Dense(hidden_size, hidden_size, elu) for _ in 1:num_layers]...,    # Hidden layers
        Dense(hidden_size, output_size),                                    # Output layer
    )
    return model
end

# Create model: 1 input, 3 hidden layers of size 30, 1 output
model = FNN(1, 30, 3, 1)

# Get default random number generator
rng = Random.default_rng()

# Define training function for the FNN
function train(model, device, rng, kwargs...)
    lr = 1e-3                       # Learning rate
    n_epochs = 5_000                # Number of training epochs
    optimizer = Adam(lr)            # Use Adam optimizer
    
    # Initialize parameters and move to device
    ps, st = Lux.setup(rng, model) |> device

    vjp = AutoZygote()              # Use Zygote for automatic differentiation
    lossfn = MSELoss()              # Mean squared error loss

    # Create training state
    train_state = Training.TrainState(model, ps, st, optimizer)

    tr_acc = 0.0                    # Placeholder for accuracy (not used)
    n = size(xt, 2)                 # Number of training samples
    stime = time()                  # Track training start time

    for epoch in 1:n_epochs
        # Accumulate loss for each epoch
        loss_acc = 0.0
        for i in 1:n
            # Fetch one sample, move to device
            x, y = xt[:, i] |> device, yt[:, i] |> device
            
            # Perform one training step
            _, loss, _, train_state = Training.single_train_step!(
                vjp, lossfn, (x, y), train_state
            )
            loss_acc = loss
        end
        if epoch % 100 == 0
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss_acc
        end
    end

    # Calculate total training time
    ttime = time() - stime
    return train_state, tr_acc, ttime
end

# Start training the model on device
tstate, tr_acc, tr_time = train(model, xdev, rng)

# Compile the model forward pass for training input
forward_pass = @compile Lux.apply(
    tstate.model, xdev(xt), tstate.parameters, Lux.testmode(tstate.states)
)

# Perform forward pass and bring prediction back to CPU
y_pred = cdev(
    first(
        forward_pass(tstate.model, xdev(xt), tstate.parameters, Lux.testmode(tstate.states))
    ),
)

# Plot training result
f = Figure(size = (600, 650))
ax = Axis(f[1, 1])
lines!(ax, xt[1, :], yt[1, :], label="Data Train")       # Plot true (noisy) data
lines!(ax, xt[1, :], y_pred[1, :], label="Train Prediction")  # Plot prediction
axislegend(position = :rb)                              # Add legend
f

# Compile model forward pass for test input
forward_pass = @compile Lux.apply(
    tstate.model, xdev(x_test), tstate.parameters, Lux.testmode(tstate.states)
)

# Predict on test data
y_test_pred = cdev(
    first(
        forward_pass(tstate.model, xdev(x_test), tstate.parameters, Lux.testmode(tstate.states))
    ),
)

# Plot test prediction result
f2 = Figure(size = (600, 650))
ax2 = Axis(f2[1, 1])
lines!(ax2, x_test[1, :], y_test[1, :], label="Data Test Ground Truth")   # True signal
lines!(ax2, x_test[1, :], y_test_pred[1, :], label="Test Prediction")     # Predicted signal
axislegend(position = :rb)
f2