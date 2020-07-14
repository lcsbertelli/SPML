python mnist_raw_native.py > results/mnistfspf/mnist_native.log
python mnist_raw_DP.1.py > results/mnistfspf/mnist_dp.1_1.log
python mnist_raw_DP1.py > results/mnistfspf/mnist_dp1.log
python mnist_raw_DP2.py > results/mnistfspf/mnist_dp2.log
python mnist_raw_DP4.py > results/mnistfspf/mnist_dp4.log
python mnist_raw_DP6.py > results/mnistfspf/mnist_dp6.log
python mnist_raw_DP8.py > results/mnistfspf/mnist_dp8.log


mv Mnistmodel* mnistfspf/

python mnist_raw_inference_native.py > results/mnistfspf/mnist_inference_native.log
python mnist_raw_inference_dp.1.py > results/mnistfspf/mnist_inference_.1.log
python mnist_raw_inference_dp1.py > results/mnistfspf/mnist_inference_1.log
python mnist_raw_inference_dp2.py > results/mnistfspf/mnist_inference_2.log
python mnist_raw_inference_dp4.py > results/mnistfspf/mnist_inference_4.log
python mnist_raw_inference_dp6.py > results/mnistfspf/mnist_inference_6.log
python mnist_raw_inference_dp8.py > results/mnistfspf/mnist_inference_8.log

#python mnist_attack_native.py > results/mnisthw/mnist_attack_native.log
#python mnist_attack_DP.1.py > results/mnisthw/mnist_attack_dp.1_1.log
#python mnist_attack_DP1.py > results/mnisthw/mnist_attack_dp1.log
#python mnist_attack_DP2.py > results/mnisthw/mnist_attack_dp2.log
#python mnist_attack_DP3.py > results/mnisthw/mnist_attack_dp4.log
#python mnist_attack_DP4.py > results/mnisthw/mnist_attack_dp6.log
#python mnist_attack_DP8.py > results/mnisthw/mnist_attack_dp8.log

#mv Mnistattack* mnisthw/

#python mnist_full_native.py > results/mnistfullhw/mnist_attack_native.log
#python mnist_full_DP.1.py > results/mnistfullhw/mnist_attack_dp.1.log
#python mnist_full_DP1.py > results/mnistfullhw/mnist_attack_dp1.log
#python mnist_full_DP2.py > results/mnistfullhw/mnist_attack_dp2.log
#python mnist_full_DP4.py > results/mnistfullhw/mnist_attack_dp4.log
#python mnist_full_DP6.py > results/mnistfullhw/mnist_attack_dp6.log
#python mnist_full_DP8.py > results/mnistfullhw/mnist_attack_dp8.log

#mv Mnistmodel* mnistfullhw/

#python mnist_attack_full_native.py
#python mnist_attack_full_DP.1.py
#python mnist_attack_full_DP1.py
#python mnist_attack_full_DP2.py
#python mnist_attack_full_DP4.py
#python mnist_attack_full_DP6.py
#python mnist_attack_full_DP8.py

#mv Mnistattack* mnistfullhw/

