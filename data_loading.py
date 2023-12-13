import pandas as pd
import io
from torch.utils.data import RandomSampler, DataLoader


# loading one line of data only, file is too large :/
def load_train_data():
    f_train = open('data/donnees_appr_dev/train.xml', "r")
    train_xml = f_train.read()
    train = pd.read_xml(io.StringIO(train_xml))
    return train


def load_dev_data():
    f_dev = open('data/donnees_appr_dev/dev.xml', "r")
    dev_xml = f_dev.read()
    dev = pd.read_xml(io.StringIO(dev_xml))
    return dev




train_df = load_train_data()[["note", "commentaire"]]
print(train_df.tail())

dev_df = load_dev_data()[["note", "commentaire"]]
print(dev_df.tail())

train_sample_size = 30271
# diviseurs 2, 11, 22, 30271, 60542, 332981
dev_sample_size = 20080
# diviseurs 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 200, 251, 400, 502, 1004, 1255, 2008, 2510, 4016, 5020, 6275, 10040, 12550, 20080, 25100, 50200,
