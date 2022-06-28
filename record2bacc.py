import os

def parse_dict(logpath, keyword):
    '''
        Args:
            - logpath: the path of logfile
            - keyword: the keyword to find in the file

        Returns the parsed dictionary
    '''
    d = {}
    with open(logpath, mode='r') as f:
        rows = f.readlines()
        target_word = keyword
        for row in rows:
            pos = row.find(target_word)  # row must have word "Train Loss: "
            if pos != -1:
                # find dataset name, as key
                dataset_name = row[:row.find("|")].strip()
                if dataset_name not in d:
                    d[dataset_name] = []

                pos += len(target_word)
                row = row[pos:]  # strip all the things before the loss

                spacepos = row.find(' ')  # find the space after the number
                if spacepos != -1:
                    row = row[:spacepos]  # strip all things after the loss

                loss = float(row)
                d[dataset_name].append(loss)
    return d

# path = 'logs_fedbn_fedprox_fedavg/DigitModel'
path = 'log_choke'
for root, dirs, files in os.walk(path):
    for file in files:
        words = file.split()
        _file = os.path.join(root, file)
        d = parse_dict(_file, 'Test  Acc: ')
        bacc = max(d['server'])
        print(f'{file}\t\tbest acc: {bacc}')

