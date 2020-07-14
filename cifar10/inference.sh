echo "" > results/cifarhw/Cifar10_inference_native.log
echo "" > results/cifarhw/Cifar10_inference_.1.log
echo "" > results/cifarhw/Cifar10_inference_1.log
echo "" > results/cifarhw/Cifar10_inference_2.log
echo "" > results/cifarhw/Cifar10_inference_4.log
echo "" > results/cifarhw/Cifar10_inference_6.log
echo "" > results/cifarhw/Cifar10_inference_8.log

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_native.log
	python Cifar10_inference_native.py >> results/cifarhw/Cifar10_inference_native.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_.1.log
	python Cifar10_inference.1.py >> results/cifarhw/Cifar10_inference_.1.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_1.log
	python Cifar10_inference1.py >> results/cifarhw/Cifar10_inference_1.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_2.log
	python Cifar10_inference2.py >> results/cifarhw/Cifar10_inference_2.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_4.log
	python Cifar10_inference4.py >> results/cifarhw/Cifar10_inference_4.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_6.log
	python Cifar10_inference6.py >> results/cifarhw/Cifar10_inference_6.log
done

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
  echo "=====================ITERATION $VARIABLE==========================" >> results/cifarhw/Cifar10_inference_8.log
	python Cifar10_inference8.py >> results/cifarhw/Cifar10_inference_8.log
done
