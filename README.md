# FL-NonIID-algorithms-and-benchmark
Group Project for EI314, Non-IID problem in Federated Learning

## Benchmark

Project include 6 kind of algorithm: FedBN、FedProx、FedAvg、PerFedAvg、PFedMe、Scaffold and Moon.

#### Run the benchmark

##### FedBN_label_weighted.py

run FedBN_label_weighted.py to test FedBN、FedProx and FedAvg in different skew condition by using:

```
python FedBN_label_weighted.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode ['fedavg', 'fedprox', 'fedbn'] --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] 
```

Also you can decide the client number、batch size、global epoch、local epoch and skew parameters through parser.  Detailed settings sees in the code file.



##### PerFedAvg_PFedMe.py

run PerFedAvg_PFedMe.py to test PerFedAvg、PFedMe in different skew condition by using:

```
python PerFedAvg_PFedMe.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode ['perfedavg', 'pfedme'] --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] 
```

##### Scaffold.py

run Scaffold.py to test Scaffold in different skew condition by using:

```
python Scaffold.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode scaffold --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] 
```

##### Moon.py

run Moon.py to test Moon in different skew condition by using:

```
python PerFedAvg_PFedMe.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode moon --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] 
```

#### Skew details

The realization of all the skews are in skew.py



#### Logs of benchmark

All the log files are in "./logs/DigitModel" folder

use the record2plot.py to generated the plot of benchmark

use record2bacc.py to generated the best accuracy of different logs.

## Label Weighted FedBN



Run FedBN_label_weighted.py to test our Label-Weighted-FedBN in different skew condition by using:

```
python FedBN_label_weighted.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode ['fedavg', 'fedprox', 'fedbn'] --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] --label
```



## Choking When Aggregation

Run FedBN_label_weighted.py to test our Choking method for FedAvg and FedProx in different skew condition by using:

```
python FedBN_label_weighted.py --dataset ['svhn', 'cifar10', 'mnist', 'kmnist'] --mode ['fedavg', 'fedprox'] --skew ['none', 'quantity', 'feat_filter', 'feat_noise', 'label_across', 'label_within'] --choke
```

The logs of choking method and its comparation are in './log_choke/' folder
