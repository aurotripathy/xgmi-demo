#HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_MIN_NCHANNELS=4 python3.6 moded_micro_benchmarking_pytorch.py --network ResNeXt101_32C_48d --dataparallel --device_ids=0,1,2,3,4,5,6,7 --iterations 10 --batch-size 128
HSA_FORCE_FINE_GRAIN_PCIE=1 python3.6 moded_micro_benchmarking_pytorch.py --network ResNeXt101_32C_48d --dataparallel --device_ids=0,1,2,3,4,5,6,7 --iterations 10 --batch-size 128
HSA_FORCE_FINE_GRAIN_PCIE=1 python3.6 moded_micro_benchmarking_pytorch.py --network ResNeXt101_32C_48d --dataparallel --device_ids=0,1,2,3,4,5,6,7 --iterations 10 --batch-size 128
HSA_FORCE_FINE_GRAIN_PCIE=1 python3.6 moded_micro_benchmarking_pytorch.py --network ResNeXt101_32C_48d --dataparallel --device_ids=0,1,2,3,4,5,6,7 --iterations 10 --batch-size 128


