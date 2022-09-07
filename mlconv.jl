using Plots, Interact, JLD, Flux
using Flux: Dense, Chain, onehotbatch, argmax, crossentropy, throttle, mse, onecold, shuffle, relu, sigmoid
using Flux.Data: DataLoader
using IterTools: ncycle, partition
using Plots, Interact, ProgressMeter, LinearAlgebra
using Images
using Flux, Zygote, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle, mse, flatten
using Base.Iterators: repeated, partition
using Random:randperm
using ImageFiltering, Images, Interact, Plots

function matrix2convtensor(X::AbstractArray)

    @assert  2 <= ndims(X) <=4
    if ndims(X) == 2
        new_size = (size(X)...,1,1)
        return reshape(X, new_size)

    elseif ndims(X) == 3
        new_size = (size(X)...,1)
        return reshape(X, new_size)
    end

end

function shouldFlip()
    choice = mod((rand(1:10)), 2)

    if choice == 1
        return true
    end
    return false

end

function createData(amt::Integer = 1000)

    imageData = Gray.(zeros(64,64, amt))

    data = zeros(64,64,amt)

    labels = zeros(1000)

    for i in 1:amt
        file = "/Users/Thomas/Desktop/ML-research/dogs/"
        if i < 10
            stringToAppend = "dogs_0000" * string(i) * ".jpg"
        elseif i < 100
            stringToAppend = "dogs_000" * string(i) * ".jpg"
        elseif i < 1000
            stringToAppend = "dogs_00" * string(i) * ".jpg"
        else
            stringToAppend = "dogs_0" * string(i) * ".jpg"
        end
        file = file * stringToAppend

        println(file)
        x = load(file)

        gray_image = Gray.(x)

        raw = imresize(gray_image,64,64)

        sf = shouldFlip()
        if sf == true
            raw = imrotate(raw, pi)
        end

        img = Float64.(raw)

        imageData[:,:,i] = raw
        data[:,:,i] = img

        sf = sf ? 1 : 0

        labels[i] = sf

    end
    return imageData, data, labels'
end

function generate_idxs(sample_idxs)

    idxs = zeros(10)
    sz = size(sample_idxs)[1]
    for i in 1:10
        x = rand(1:sz)
        idxs[i] = sample_idxs[x]
    end
    return Int64.(idxs)
end

function visualize_output(model, sample_idxs, dataCube, whichLabels)
    plots = []
    kwargs = (
        :clim => (-0.2, 0.2),
        :color => :grays,
        :aspect_ratio => :equal,
        :ticks => :off,
        :colorbar => :false,
        :showaxis => :false,
        :grid => false
    )

    for idx in sample_idxs
        print(idx)
        print("\n")
        title_string = "unflipped"
        if whichLabels[idx] == 1
            title_string = "flipped"
        end
        push!(plots, heatmap(
                dataCube[:,:,idx];
                title=title_string,
                kwargs...
            )
        )
    end
    plot(plots...; layout=(2,5), size=(800, 350))
end

function filter2weight(filter)
    W = zeros(3,3)

    if filter == "identity"
        return W = [0.0 0.0 0.0; 0.0 1.0 0.0 ; 0.0 0.0 0.0]

    end
    if filter == "sharpen"
        return W = [ 0 -1 0; -1 5 -1; 0 -1 0]/5
    end

    if filter == "gaussian"
        return W = [ 1 2 1; 2 4 2; 1 2 1]/16
    end

    if filter == "edge detection"
        return W = [ -1.0 -1.0 -1.0; -1.0 8.0 -1.0; -1.0  -1.0 -1.0]
    end

     if filter == "random"
        return W = rand(3,3)
    end

    return W
end

amt = 1000

#create training data in batches of 1000
rawData, picturesOfDogs, labels = createData(amt)
rawData2, picturesOfDogs2, labels2 = createData(amt)
rawData3, picturesOfDogs3, labels3 = createData(amt)
rawData4, picturesOfDogs4, labels4 = createData(amt)
rawData5, picturesOfDogs5, labels5 = createData(amt)

#merge raw image data together
rawData = cat(rawData,rawData2, dims = 3)
rawData = cat(rawData,rawData3, dims = 3)
rawData = cat(rawData,rawData4, dims = 3)
rawData = cat(rawData,rawData5, dims = 3)

#merge training data together into a conv. tensor
picturesOfDogs = cat(picturesOfDogs,picturesOfDogs2, dims = 3)
picturesOfDogs = cat(picturesOfDogs,picturesOfDogs3, dims = 3)
picturesOfDogs = cat(picturesOfDogs,picturesOfDogs4, dims = 3)
picturesOfDogs = cat(picturesOfDogs,picturesOfDogs5, dims = 3)

#merging labels, adjusting var. type and converting to 5000x1.
labels = hcat(labels, labels2)
labels = hcat(labels, labels3)
labels = hcat(labels, labels4)
labels = hcat(labels, labels5)
labels = Int64.(labels')

testData, testDogs, testLabels = createData(amt)
testData2, testDogs2, testLabels2 = createData(amt)
testData3, testDogs3, testLabels3 = createData(amt)

# merge testing data together
testData = cat(testData,testData2, dims = 3)
testData = cat(testData,testData3, dims = 3)

testDogs = cat(testDogs,testDogs2, dims = 3)
testDogs = cat(testDogs,testDogs3, dims = 3)

testLabels = hcat(testLabels, testLabels2)
testLabels = hcat(testLabels, testLabels3)

dogs_x_train = picturesOfDogs
dogs_y_train = onehotbatch(labels, 0:1)

dogs_x_test = testDogs
dogs_y_test = onehotbatch(testLabels', 0:1)

unpermuted_conv_training_tensor = Float32.(reshape(picturesOfDogs, (size(picturesOfDogs)...,1)))
training_tensor = permutedims(unpermuted_conv_training_tensor,[1,2,4,3]);

unpermuted_conv_testing_tensor = Float32.(reshape(testDogs, (size(testDogs)...,1)))
testing_tensor = permutedims(unpermuted_conv_testing_tensor,[1,2,4,3]);

train_img = picturesOfDogs[:,:,1]
train_img_tensor = matrix2convtensor(train_img)
prod(size(conv_deep_model(train_img_tensor)))

conv_model = Chain(
    Conv((3, 3), 1 => 9),
    MaxPool((2,2)) ,
    Conv((3,3) , 9 => 16),
    MaxPool((2,2))
)
deep_model = Chain(x -> reshape(x, :, size(x, 4)),
    Dense(3136, 10, relu),
    Dense(10, 2 ),
    softmax
)

conv_deep_model = Chain(conv_model,deep_model)
loss(x,y) = crossentropy(conv_deep_model(x),y)

accuracy(x,y) = mean(onecold(x) .== onecold(y))

train_accuracy = accuracy(conv_deep_model(training_tensor),dogs_y_train)

batch_size = 60
opt = ADAM()

for iters =  1 : 1000

    batch_idxs = randperm(size(training_tensor,4))[1:batch_size]
    x_train_batch_tensor = training_tensor[:,:,[1],batch_idxs]
    train_set = (x_train_batch_tensor, dogs_y_train[:,batch_idxs])

    Flux.train!(loss, params(conv_deep_model), [train_set], opt)

    if iters % 50 == 0

        train_loss = loss(train_set[1],train_set[2])
        batch_idxs = randperm(size(testing_tensor,4))[1:1000]
        test_loss = loss(testing_tensor[:,:,[1],batch_idxs],dogs_y_test[:,batch_idxs])
        test_accuracy = accuracy(conv_deep_model(testing_tensor),dogs_y_test)
        println("Batch training loss is $(train_loss), Test loss is $(test_loss), Test accuracy is $(test_accuracy)")

    end

end
