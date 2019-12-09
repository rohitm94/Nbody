SRC=gpu_opt.cu
EXE=gpu_opt

K=65536
for i in {1..5}
do
    ./$EXE $K
    K=$(($K*2))
done

