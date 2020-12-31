import torch


class Model(torch.nn.Module):
    def __init__(self):


        super().__init__()
        self.layer_one = torch.nn.Linear(64, 256)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(256, 256)
        self.activation_two = torch.nn.ReLU()

        self.shape_outputs = torch.nn.Linear(16384, 2)


    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = buffer.flatten(start_dim=1)
        return self.shape_outputs(buffer)

inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)
inputs, outputs

model = Model()
test_results = model(inputs)
test_results

loss_fn = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=.01)

EPOCHS = 10000
for i in range(EPOCHS):
    optim.zero_grad()
    results = model(inputs)
    loss = loss_fn(results, outputs)
    loss.backward()
    optim.step()
    gradients = 0.0
    for parameter in model.parameters():
        gradients += parameter.grad.sum()
    if abs(gradients) <= .0001:
        print(gradients)
        print("grardients vanished at iteration {0}".format(i))
        break
print(model(inputs), outputs)
