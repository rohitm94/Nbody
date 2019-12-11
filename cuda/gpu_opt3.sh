SRC=gpu_opt3.cu
EXE=gpu_opt3

echo $EXE

K=65536
for i in {1..4}
do
    ./$EXE $K
    K=$(($K*2))
done
