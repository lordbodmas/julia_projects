using DataFrames
using CSVFiles
using Flux # ML package that allows us to leverage the differentialble nature of Julia

W = rand(2, 5)
b = rand(2)

# let us define a simple linear regression formula
predict(x) = W*x .+ b # `.` allows Julia to understand the vector nature of the operation

# next, let us define our loss function

function loss(x, y)
    ŷ = predict(x)
    sum((y .- ŷ).^2) # in julia, no need to use the return statement
end

# dummy data
x, y = rand(5), rand(2) # Dummy data

println("Loss before any weights update: $(loss(x, y))")


# gradient descent
gs = gradient(() -> loss(x, y), params(W, b)) # params(W, b) tells Flux that these are the params to use

w̄ = gs[W] # gradients

W .-= 0.1 .* w̄

println("Loss after single weights update: $(loss(x, y))")
