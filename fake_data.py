import torch

class fake_data(object):


    def __init__(self, n, lbs, str_cond=True):
        # https://wiki.python.org/moin/Generators
        print("Fake data hacked for gpu SD")
        self.n = n
        self.lbs = lbs
        self.num = 0
        self.str_cond = str_cond


    def __iter__(self):
        return self


    # Python 3 compatibility
    def __next__(self):
        return self.next()


    def next(self):
        if self.num < self.n/self.lbs:
            self.num += 1
            txt_return = ['asdf']*self.lbs if self.str_cond else torch.randint(low=0, high=10000, size=(self.lbs, 77))
            return torch.zeros(self.lbs, 3, 512, 512).detach(), txt_return
        

        raise StopIteration()


if __name__ == '__main__':
    fake_ds = fake_data(16, 4)
    for x in fake_ds:
        print(x)
        print(x[0].shape)
    print(x[1].shape)

