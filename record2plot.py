import argparse
import os
import matplotlib.pyplot as plt

# use this to parse train_loss, other loss is meaning less to be analyzed
def parse_dict(dict, logpath, keyword):
    '''
        Args:
            - dict: the HashTable to save the history
                - key: dataset name
                - value: list of history
            - logpath: the path of logfile
            - keyword: the keyword to find in the file
    '''
    with open(logpath, mode='r') as f:
        rows = f.readlines()
        target_word = keyword
        for row in rows:
            pos = row.find(target_word)  # row must have word "Train Loss: "
            if pos != -1:
                # find dataset name, as key
                dataset_name = row[:row.find("|")].strip()
                if dataset_name not in dict:
                    dict[dataset_name] = []

                pos += len(target_word)
                row = row[pos:]  # strip all the things before the loss

                spacepos = row.find(' ')  # find the space after the number
                if spacepos != -1:
                    row = row[:spacepos]  # strip all things after the loss

                loss = float(row)
                dict[dataset_name].append(loss)

def draw_plot(x_hist,
              y_hists,
              labels,
              title,
              x_label,
              y_label,
              name):
    plt.cla()  # clean all plots before
    for i, y_hist in enumerate(y_hists):
        plt.plot(x_hist, y_hist, label=labels[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, name))


'''
    In logpos folder, you should have fedbn.log, fedprox.log, fedavg.log
'''
parser = argparse.ArgumentParser()
parser.add_argument("--logpos", type=str, default="./checkpoints")
parser.add_argument("--out_dir", type=str, default="./fig_folder")
args = parser.parse_args()

folder = args.logpos

loss_history_fedbn = {}
loss_history_fedprox = {}
loss_history_fedavg = {}

tacc_history_fedbn = {}
tacc_history_fedprox = {}
tacc_history_fedavg = {}

logfile_fedbn = os.path.join(folder, "fedbn.log")
logfile_fedprox = os.path.join(folder, "fedprox.log")
logfile_fedavg = os.path.join(folder, "fedavg.log")

parse_dict(loss_history_fedavg, logfile_fedavg, "Train Loss: ")
parse_dict(loss_history_fedprox, logfile_fedprox, "Train Loss: ")
parse_dict(loss_history_fedbn, logfile_fedbn, "Train Loss: ")

parse_dict(tacc_history_fedavg, logfile_fedavg, "Test  Acc: ")
parse_dict(tacc_history_fedprox, logfile_fedprox, "Test  Acc: ")
parse_dict(tacc_history_fedbn, logfile_fedbn, "Test  Acc: ")

for dname in loss_history_fedbn.keys():
    # save the plot named dname
    loss_fedavg = loss_history_fedavg[dname]
    loss_fedprox = loss_history_fedprox[dname]
    loss_fedbn = loss_history_fedbn[dname]

    tacc_fedavg = tacc_history_fedavg[dname]
    tacc_fedprox = tacc_history_fedprox[dname]
    tacc_fedbn = tacc_history_fedbn[dname]

    nepochs = range(1, len(loss_fedbn) + 1)

    draw_plot(x_hist=nepochs,
              y_hists=[loss_fedbn,
                       loss_fedprox,
                       loss_fedavg],
              labels=["FedBN", "FedProx", "FedAvg"],
              title=f"Loss History on {dname}",
              x_label="Epochs",
              y_label="Loss",
              name=f'Loss_{dname}.png')
    
    draw_plot(x_hist=nepochs,
              y_hists=[tacc_fedbn,
                       tacc_fedprox,
                       tacc_fedavg],
              labels=["FedBN", "FedProx", "FedAvg"],
              title=f"Test Accuracy History on {dname}",
              x_label="Epochs",
              y_label="Accuracy (%)",
              name=f'Tacc_{dname}.png')
