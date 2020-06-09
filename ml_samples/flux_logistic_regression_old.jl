using Flux, Statistics
using Flux: Params, gradient, logitcrossentropy
using Flux.Optimise: update!
using Flux.Data: Iris
using Parameters: @with_kw

"""
This code is a modification of the below example for Iris classification
https://github.com/FluxML/model-zoo/blob/master/other/housing/housing.jl
"""

# Struct to define hyperparameters
@with_kw mutable struct Args
    lr::Float64 = 0.1		# learning rate
    split_ratio::Float64 = 0.1	# Train Test split ratio, define percentage of data to be used as Test data
    epochs::Int = 200 # number of updates to the weights
end


function normalise_data(x)
    (x .- mean(x, dtims = 2)) ./ std(x, dims = 2)
end


function split_data(x, y, split_ratio)
    split_index = floor(Int, size(x,1)*split_ratio)
    x_train = x[1:split_index, :]
    y_train = y[1:split_index, :]
    x_test = x[split_index+1:size(x,1), :]
    y_test = y[split_index+1:size(x,1), :]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data, test_data
end


function load_iris_data(args)
    """
    Loads the Iris classification data set via flux. Splits into train and test
    examples
    """
    features = Iris.features()
    features = permutedims(features) # columns and rows need to be swapped
    labels = Iris.labels()

    train_data, test_data = split_data(features, labels, args.split_ratio)
    return train_data, test_data
end


mutable struct model
    W::AbstractArray
    b::AbstractVector
end

# prediction function
function predict(x, model)
    println(size(model.W))
    println(size(model.b))
    model.W*x .+ model.b
end
# predict(x, model) = model.W*x .+ model.b

# accuracy function
accuracy(x, y, model) = mean(onecold(predict(x, model)) .== onecold(y))


function train(; kws...)
    # initialise the hyperparameters
    args = Args()
    (x_train,y_train),(x_test,y_test) = load_iris_data(args)

    # instantiate model with random initial weights
    m = model((randn((10, 4))),[0.])

    # define our loss function
    loss(x, y) = accuracy(x, y, m) #logitcrossentropy(predict(x, m), y)

    println("initial loss of $(loss(x_train, y_train))")
    println("here")
    # let us train η = args.lr
    θ = params([m.W, m.b])
    for i = 1: args.epochs
        g = gradient(() -> loss(x_train, y_train), θ)
        for x in θ
            update!(x, -g[x]*η)
        end

        if i % 100 == 0
            @show loss(x_train , y_train)
        end
    end

    # evaluate on test data
    test_accuracy = accuracy(X_test, y_test, m)
    println(test_accuracy)
end



#####  call the functions below #####
train()
