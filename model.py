import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
from image_handling import load_data

def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, gpu):
    trainloader, testloader, validloader = load_data(data_dir)
    
    model = torch.hub.load('pytorch/vision', arch, pretrained=True)
    
    for p in model.parameters():
        p.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier

    device = torch.device("cuda" if gpu else "cpu")

    model.to(device);

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    running_loss = 0

    curr_step = 0
    printing_step = 5

    for epoch in range(epochs):

        for inputs, labels in trainloader:

            curr_step += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if curr_step%printing_step == 0:

                val_loss = 0
                val_accuracy = 0
                model.eval()

                with torch.no_grad():

                    for inputs_1, labels_1 in validloader:

                        inputs_1, labels_1 = inputs_1.to(device), labels_1.to(device)

                        logps = model.forward(inputs_1)
                        batch_loss = criterion(logps, labels_1)

                        val_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_1.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/printing_step:.3f}.. "
                          f"Validation loss: {val_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {val_accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()
                
    model.eval()