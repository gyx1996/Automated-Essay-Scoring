import numpy as np
##################
# ???
from pointer_network import PointerNetwork
# ???
####################
import torch
MAX_EPOCH = 100000


#############################
# ?????construct your model
shiyan = 3

if shiyan == 1:
    senlen = 5
elif shiyan == 2:
    senlen = 10
else:
    senlen = 5

model = PointerNetwork(output_size=senlen)

# model = torch.load('model/1-6000.pkl')
# ?????
#############################


batch_size = 256

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss_fun = torch.nn.CrossEntropyLoss()


def getdata(shiyan=shiyan, batch_size=batch_size):
    if shiyan == 1:
        high = 100
        senlen = 5
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 2:
        high = 100
        senlen = 10
        x = np.array([np.random.choice(range(high), senlen, replace=False) for _ in range(batch_size)])
        y = np.argsort(x)
    elif shiyan == 3:
        senlen = 5
        x = np.array([np.random.random(senlen) for _ in range(batch_size)])
        y = np.argsort(x)
    return x, y


def evaluate():
    accuracy_sum = 0.0
    for i in range(300):
        test_x, test_y = getdata(shiyan=shiyan)
        ###############################
        # ????
        test_x = torch.FloatTensor(test_x).transpose(0, 1)
        test_y = torch.FloatTensor(test_y).transpose(0, 1)
        predictions = model(test_x, test_y, training=False)
        accuracy = sum([1 if torch.equal(pred.data, y.data) else 0
                        for pred, y in zip(predictions, test_y)])
        # compute prediction ,and then get the accuracy
        #############################
        accuracy_sum += accuracy
    print('accuracy is ', accuracy_sum / (batch_size * 300.0))
    if accuracy_sum / (batch_size * 300.0) > 0.95:
        exit(0)


for epoch in range(MAX_EPOCH):
    train_x, train_y = getdata(shiyan=shiyan)

    ############################
    # compute the  prediction
    train_x = torch.FloatTensor(train_x).transpose(0, 1)
    train_y = torch.FloatTensor(train_y).transpose(0, 1)
    prediction = model(train_x, train_y, training=True).view(-1, senlen)
    train_y = train_y.contiguous().view(-1)
    ###########################
    loss = loss_fun(prediction, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, ' \t loss is \t', loss.item())
    print(loss)

    if epoch % 200 == 0 and epoch != 0:
        print(epoch, ' \t loss is \t', loss.item())
        print(loss)
        torch.save(model, 'model/shiyan' + str(shiyan) + '-' + str(epoch) + '.pkl')
        evaluate()

    if epoch % 2000 == 0 and epoch != 0:
        evaluate()
