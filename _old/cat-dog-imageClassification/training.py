import torch
import matplotlib.pyplot as plt


def train(model, criterion, optimizer, trainloader, validationloader, epochs=5, device="cpu"):
    # we will need these to display the loss graph at the end of the training
    train_losses, validation_losses = [], []

    for e in range(epochs):
        # set model to training mode
        model.train()

        training_loss = 0
        for images, labels in trainloader:
            # cast images, labels to the preferred device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # zero gradients
            log_ps = model(images)  # get the output (log probability)
            loss = criterion(log_ps, labels)  # calculate our loss
            loss.backward()  # calculate the gradients
            optimizer.step()  # and perform a gradient descent step

            training_loss += loss

        else:
            training_loss /= len(trainloader)

            # set model to evaluation mode
            model.eval()

            validation_loss = 0
            validation_accuracy = 0

            with torch.no_grad():
                # cast images, labels to the preferred device
                for images, labels in validationloader:
                    # cast images, labels to the preferred device
                    images, labels = images.to(device), labels.to(device)

                    log_ps = model(images)  # get the output (log probability)
                    validation_loss += criterion(log_ps, labels)  # calculate the validation loss

                    ps = log_ps.exp()  # get the probabilities
                    top_ps, top_class = ps.topk(1, dim=1)  # take the top class prediction and the probability of it
                    equals = top_class == labels.view(
                        *top_class.shape)  # see which top class predictions match to the labels
                    validation_accuracy += equals.type(
                        torch.FloatTensor).mean()  # and calculate the validation accuracy

            validation_loss /= len(validationloader)
            validation_accuracy /= len(validationloader)

            train_losses.append(training_loss)
            validation_losses.append(validation_loss)

            print("Epoch %d, training loss %f, validation loss %f, validation accuracy %f" %
                  (e, training_loss.item(), validation_loss.item(), validation_accuracy.item()))

    # plot the losses after training to get intuitions
    plt.plot(train_losses, label="Training loss")
    plt.plot(validation_losses, label="Validation loss")
    plt.legend(frameon=False)
    plt.show()


def test(model, criterion, testloader, device="cpu"):
    # set model to evaluation mode
    model.eval()

    test_loss = 0
    test_accuracy = 0

    for images, labels in testloader:
        # cast images, labels to the preferred device
        images, labels = images.to(device), labels.to(device)

        log_ps = model(images)  # get the output (log probability)
        test_loss += criterion(log_ps, labels)  # calculate the test loss

        ps = log_ps.exp()  # get the probabilities
        top_ps, top_class = ps.topk(1, dim=1)  # take the top class prediction and the probability of it
        equals = top_class == labels.view(*top_class.shape)  # see which top class predictions match to the labels
        test_accuracy += equals.type(torch.FloatTensor).mean()  # and calculate the test accuracy

    test_loss /= len(testloader)
    test_accuracy /= len(testloader)

    print("Test loss %f, test accuracy %f" % (test_loss.item(), test_accuracy.item()))
