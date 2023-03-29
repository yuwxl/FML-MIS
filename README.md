### FML-MIS: A Scheme of Privacy Protection and Model Generalization for Medical Images Segmentation via Federated Meta-learning

##### 0. Environment configuration

> python  v 3.9.9 
> PyTorch 1.10.0	Python 3.8	Cuda 11.3
> GPU	RTX 3090 * 1 24GB
> CPU	7 Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz	32GB

##### 1. dataset

```
for train
├── dataset
      ├── client 1
      	├── data_npy
      		├── sample1.npy, sample2.npy, xxxx  (384 * 384 * 5)
         ├── freq_amp_npy
            	├── amp_sample1.npy, amp_sample2.npy, xxxx
      ├── client xxx
      ├── client xxx    	
```

```
leave-one-domain-test
├── dataset
     ├── client 1
     	├── data_npy
     		├── sample1.npy, sample2.npy, xxxx  (384 * 384 * 5)
```


```
raw data   
├──FundusData
 	├──Domain		
     	├── Domin1
     		├── xxx.jpg
     	├── Dominxxx
 	├──Domain_mask
 		├── Domain1_mask
 			├── xxx.jpg
 		├── Domainxxx_mask   
```

##### 2. Train ( 1 server,3 clients)

```
python server.py --ip 127.0.0.1 --port 3002 --world_size 4 --round 100
python client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 1 --client_idx 1 --dp_mechanism Gaussian
python client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 2 --client_idx 2 --dp_mechanism Gaussian
python client.py --ip 127.0.0.1 --port 3002 --world_size 4 --rank 3 --client_idx 3 --dp_mechanism Gaussian
```

##### 3. Acknowledgement

Some of the code is adapted from [FedDG](https://github.com/liuquande/FedDG-ELCFS)、 [FDA](https://github.com/YanchaoYang/FDA) and [Fedlab](https://github.com/SMILELab-FL/FedLab). 

