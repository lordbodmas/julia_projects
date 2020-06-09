using Flux, Statistics
using Flux: Params, gradient, logitcrossentropy, onehotbatch, normalise, Dense, onecold
using Flux.Optimise: update!
using Flux.Data: Iris
using Parameters: @with_kw

"""
This code is from
https://github.com/FluxML/model-zoo/blob/master/other/iris/iris.jl
"""

# Struct to define hyperparameters
@with_kw mutable struct Args
    lr::Float64 = 0.1		# learning rate
    epochs::Int = 200 # number of updates to the weights
end


function load_iris_data(args)
    """
    Loads the Iris classification data set via flux. Splits into train and test
    examples
    """
    features = Iris.features()
    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)
    #features = permutedims(features) # columns and rows need to be swapped
    labels = Iris.labels()

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    # Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:150 ; 2:3:150]

    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, 3:3:150]
    y_test = onehot_labels[:, 3:3:150]

    #repeat the data `args.epochs` times
    train_data = Iterators.repeated((X_train, y_train), args.epochs)
    test_data = (X_test,y_test)

    return train_data, test_data

end

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:3)
    y * transpose(ŷ)
end

# accuracy function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))


function train(; kws...)
    # initialise the hyperparameters
    args = Args()
    train_data, test_data = load_iris_data(args)

    model = Chain(Dense(4, 3)) # 4 features, 3 classes for iris

    # define our loss function
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, params(model), train_data, optimiser)

    return model, test_data
end


function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.8

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
end


#####  call the functions below #####
model, test_data = train()
test(model, test_data)
