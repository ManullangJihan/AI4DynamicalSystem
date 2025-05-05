# AI For Dynamical System using Julia

This is a tutorial-style repository showcasing resource-efficient, physics-informed and science knowledge based neural networks in Julia.

## 📚 Model Overview

| ✅ Status | Model                         | Topic                                | Video       | Code                             |
|----------|-------------------------------|--------------------------------------|-------------|----------------------------------|
| ✅       | Time-Stepper Neural Network    | Predicting time-evolution dynamics   | [YouTube](https://www.youtube.com/watch?v=OIXOA6Y7z5w&t=1s) | `examples/01_TimeStepperNN.jl`  |
| ✅       | Flow-Map NN         | Predicting Lorenz Equation using Flow-Map NN     |  | `examples/02_FlowMapNN.jl`      |
| 🔜       | Physics-Informed Neural Networks (PINNs), UDEs | Learning differential dynamics from data | *Coming soon* | *In progress*                  |

## How to Use

1. Clone the repo
2. Open Julia and run:
```julia
import Pkg; Pkg.activate("."); Pkg.instantiate()
