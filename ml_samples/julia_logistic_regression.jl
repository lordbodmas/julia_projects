using Statistics
using Flux.Data: Iris

"""
Code is modified from https://towardsdatascience.com/julia-for-data-science-regularized-logistic-regression-667857a7f0ce
"""


normalise_features(x) = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

# sigmoid activation function
sigmoid_activation(z) = 1 ./ (1 .+ exp.(.-z))


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


function load_iris_data()
    """
    Loads the Iris classification data set via flux. Splits into train and test
    examples
    """
    features = Iris.features()
    features = permutedims(features) # columns and rows need to be swapped
    labels = Iris.labels()

    encoded_lookup = Dict(enumerate(sort(unique(labels)))) # wrong order. need to do equiv of {v: i for i,v in enumerate(values)}


    onehot_labels = permutedims(onehotbatch(labels, klasses))

    train_data, test_data = split_data(features, onehot_labels, 0.1)# args.split_ratio)
    return train_data, test_data
end

function cost(X, y, θ)
    """
    Calculate the cost function for features X, labels y given some weights θ
    """
    m = length(y)

    # Sigmoid predictions at current batch
    h = sigmoid_activation(X * θ)

    # Cost(hθ(x), y) = -ylog(hθ(x)) - (1 - y)log(hθ(x))

    # left side of the cost function
    positive_class_cost = ((-y)' * log.(h))

    # right side of the cost function
    negative_class_cost = ((1 .- y)' * log.(1 .- h))

    # \bf
    𝐉 = (1/m) * (positive_class_cost - negative_class_cost)
    # X' is adjugate matrix
    ∇𝐉 = (1/m) * (X') * (h-y)
    return (𝐉, ∇𝐉)
end


function logistic_regression_sgd(X, y, fit_intercept=true, η=0.01, max_iter=1000)
    """
    This function uses gradient descent to search for the weights
    that minimises the logit cost function.
    A tuple with learned weights vector (θ) and the cost vector (𝐉)
    are returned.
    """

    # Initialize some useful values
    m = length(y); # number of training examples

    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X # Assume user added constants
    end

    # Use the number of features to initialise the theta θ vector
    n = size(X)[2]
    θ = zeros(n)

    # Initialise the cost vector based on the number of iterations
    𝐉 = zeros(max_iter)

    for iter in range(1, stop=max_iter)

        # Calcaluate the cost and gradient (∇𝐉) for each iter
        𝐉[iter], ∇𝐉 = cost(X, y, θ)

        # Update θ using gradients (∇𝐉) for direction and (η) for the magnitude of steps in that direction
        θ = θ - (η * ∇𝐉)
    end

    return (θ, 𝐉)
end


"""
This function uses the learned weights (θ) to make new predictions.
Predicted probabilities are returned.
"""
function predict_proba(X, θ, fit_intercept=true)
    m = size(X)[1]

    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end

    h = sigmoid(X * θ)
    return h
end


"""
This function binarizes predicted probabilities using a threshold.
Default threshold is set to 0.5
"""
function predict_class(proba, threshold=0.5)
    return proba .>= threshold
end


function train_and_evaluate()
    (X_train, y_train), (X_test, y_test) = load_iris_data()
    println(size(X_train), size(y_train), size(X_test), size(y_test))

    θ, 𝐉 = logistic_regression_sgd(X_train, y_train)
    train_score = mean(y_train .== predict_class(predict_proba(X_train, θ)));
    test_score = mean(y_test .== predict_class(predict_proba(X_test_, θ)));
    println("Training score: ", round(train_score, sigdigits=4))
    println("Testing score: ", round(test_score, sigdigits=4))

end



train_and_evaluate()
