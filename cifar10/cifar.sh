python Cifar10.py > results/cifarhw/cifar_native.log
python Cifar10eps.1.py > results/cifarhw/cifar_dp.1.log
python Cifar10eps1.py > results/cifarhw/cifar_dp1.log
python Cifar10eps2.py > results/cifarhw/cifar_dp2.log
python Cifar10eps4.py > results/cifarhw/cifar_dp4.log
python Cifar10eps6.py > results/cifarhw/cifar_dp6.log
python Cifar10eps8.py > results/cifarhw/cifar_dp8.log

mv Cifarmodel* cifarhw/

python Cifar10_inference_native.py > results/cifarhw/cifar10_inference_native.log
python Cifar10_inference.1.py > results/cifarhw/cifar10_inference_.1.log
python Cifar10_inference1.py > results/cifarhw/cifar10_inference_1.log
python Cifar10_inference2.py > results/cifarhw/cifar10_inference_2.log
python Cifar10_inference4.py > results/cifarhw/cifar10_inference_4.log
python Cifar10_inference6.py > results/cifarhw/cifar10_inference_6.log
python Cifar10_inference8.py > results/cifarhw/cifar10_inference_8.log
