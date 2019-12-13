SRC=gpu_base1.cu
EXE=gpu_base1

K=65536
for i in {1..4}
do
    ./$EXE $K
    K=$(($K*2))
done

