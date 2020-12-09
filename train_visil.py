import os
from visil import *
from compute import sim

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

gt_path = "/home/ubuntu/Desktop/CC_WEB_Video/GT/"
triplet_path = "./triplets.pickle"
label_path = "cc_web_video/cc_web_video.pickle"
weight_path = "./visil.weight"


def triplet_generator_cc():
    triplets = []
    pos = []
    for class_index in range(24):
        pos.append([])
    neg = []

    for class_index in range(24):
        gt = []
        for t in open(gt_path +"GT_" + str(class_index+1) + ".rst", "r").readlines():
            truth = t.split()
            gt.append([truth[0], truth[1]])

        for e in gt:
            if e[1] in 'ESLMV':
                # print("pos\t" + str(e))
                pos[class_index].append(e[0])
            else:
                # print("neg\t" + str(e))
                neg.append(e[0])

    features = load_rmac()

    for class_index in range(24):
        for p in range(len(pos[class_index])):
            for q in range(p+1, len(pos[class_index])):
                anchor = features[retrieve_label(features, pos[class_index][p])]
                positive = features[retrieve_label(features, pos[class_index][q])]
                negative = pick_hard_negative(anchor, neg, features)
                # print([anchor, positive, negative])
                triplets.append([anchor, positive, negative])
    return triplets

def pick_hard_negative(anchor, negatives, features):
    best_neg = None
    best_sim = 1.0
    for n in negatives:
        cur_neg = features[retrieve_label(features, n)]
        cur_sim = sim(anchor[1], cur_neg[1])
        print(str(cur_sim) + "\t" + str(best_sim))
        if best_sim > cur_sim:
            best_sim = cur_sim
            best_neg = cur_neg
    return best_neg


class FeatureDataset(Dataset):
    def __init__(self, path=triplet_path):
        if os.path.exists(path):
            self.features = pickle.load(open(label_path, 'rb'))
        else:
            self.features = triplet_generator_cc()
            with open('./triplets.pickle', 'wb') as pk_file:
                pickle.dump(self.features, pk_file, protocol = 4)
        sizes = []
        for triplet in features:
            for f in triplet:
                size.append(np.shape(f[1])[0])
        self.size = max(sizes)

        # print(self.size)

    def __getitem__(self, index):
        anchor, positive, negative = self.features[index]
        # return pad(anchor, self.size), pad(positive, self.size), pad(negative, self.size)
        # print(anchor)
        # print(np.shape(anchor))
        # print(np.shape(positive))
        # print(np.shape(negative))
        # size = [np.shape(anchor[1])[0],np.shape(positive[1])[0],np.shape(negative[1])[0]]
        return pad(anchor[1].transpose(1,0),self.size), pad(positive[1].transpose(1,0), self.size), pad(negative[1].transpose(1,0), self.size)

    def __len__(self):
        return len(self.features)



def triplet_loss(model, anchor, positive, negative, margin=1.0):
    loss = 0.
    pos_sim = np.zeros((np.shape(anchor)[0],512,512))
    neg_sim = np.zeros((np.shape(anchor)[0],512,512))
    for i in range(np.shape(anchor)[0]):
        pos_sim[i] = np.dot(anchor[i], positive[i].T)
        neg_sim[i] = np.dot(anchor[i], negative[i].T)

    pos_sim = model(pos_sim)
    neg_sim = model(neg_sim)
    loss = pos_sim - neg_sim + margin
    # print(loss)
    loss = torch.mean(torch.max(loss, torch.zeros_like(loss)))
    return loss


def train_visil(model, dataset, loss_func=triplet_loss, lr=0.001, epochs=20, batch_size=8):
    print("ViSiL Training")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    running_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(data_loader):
            anchor, positive, negative = batch

            # zero the parameter gradients
            optimizer.zero_grad()

            loss = loss_func(model, anchor, positive, negative)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                model.save_model()

model = visil(weight_path).to(device)
dataset = FeatureDataset()
train_visil(model, dataset)
