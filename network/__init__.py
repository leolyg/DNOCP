import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from train.EarlyStopping import EarlyStopping
import numpy as np
from tqdm import tqdm

from Utils import init_output


class WrapperBase():
    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError


class _PointerNetWrapper(WrapperBase):
    def __init__(self):
        self.dataset = None
        self.model = None
        self.USE_CUDA = None
        self.params = {}
        pass

    def set_dataset(self, dataset):
        self.dataset = dataset
        return self

    def set_model(self, model):
        self.model = model
        return self

    def use_cuda(self):
        self.USE_CUDA = True
        return self

    def inject(self, params):
        self.params = params
        return self

    def save_state_dict(self, folder_path):
        print("model saved in %s" % folder_path)
        torch.save(self.model.state_dict(), folder_path + "/PointerNet.pt")

    def load_state_dict(self, path="./output/PointerNet.pk"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(
            path, map_location=device), False)
        self.model.to(device)
        return self

    def train(self):
        folder_path = init_output({
            "model_meta": self.model.meta(),
            "dataset_meta": self.dataset.meta(),
            "train_meta": self.params,
            "USE_CUDA": self.USE_CUDA
        })
        dataloader = DataLoader(self.dataset,
                                batch_size=self.params["batch_size"],
                                shuffle=True,
                                num_workers=4)

        if self.USE_CUDA and torch.cuda.is_available():
            USE_CUDA = True
            print('Using GPU, %i devices.' % torch.cuda.device_count())
        else:
            USE_CUDA = False

        if self.USE_CUDA:
            self.model.cuda()
            net = torch.nn.DataParallel(
                self.model, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        CCE = torch.nn.CrossEntropyLoss()
        model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                        self.model.parameters()),
                                 lr=self.params['lr'])

        losses = []
        batch_losses = []

        patience = 50
        early_stopping = EarlyStopping(patience, verbose=True)

        for epoch in range(self.params['nof_epoch']):
            batch_loss = []
            iterator = tqdm(dataloader, unit='Batch')
            for i_batch, sample_batched in enumerate(iterator):
                iterator.set_description('Epoch %i/%i' %
                                         (epoch+1, self.params['nof_epoch']))

                train_batch = Variable(sample_batched['Points'])
                target_batch = Variable(sample_batched['Solution'])

                if USE_CUDA:
                    train_batch = train_batch.cuda()
                    target_batch = target_batch.cuda()

                o, p = self.model(train_batch)
                o = o.contiguous().view(-1, o.size()[-1])

                target_batch = target_batch.view(-1)

                loss = CCE(o, target_batch)

                losses.append(loss.item())
                batch_loss.append(loss.item())

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                iterator.set_postfix(loss='{}'.format(loss.item()))

            batch_losses.append(np.average(batch_loss))
            iterator.set_postfix(loss=np.average(batch_loss))

            early_stopping(loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                self.save_state_dict(folder_path)
                break

            if epoch % 100 == 0:
                self.save_state_dict(folder_path)

    def inference(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        infer_data = torch.stack([i['Points'] for i in data], 0).to(device)
        output, pointer = self.model(infer_data)
        return output, pointer


pointerNetWrapper = _PointerNetWrapper()
