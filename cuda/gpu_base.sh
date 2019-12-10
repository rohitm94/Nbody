SRC=gpu_base.cu
EXE=gpu_base

K=65536
for i in {1..4}
do
    ./$EXE $K
    K=$(($K*2))
done

