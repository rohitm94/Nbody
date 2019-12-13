SRC=gpu_opt2.cu
EXE=gpu_opt2

K=65536
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

