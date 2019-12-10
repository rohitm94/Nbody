SRC=gpu_opt3.cu
EXE=gpu_opt3

module load cuda
nvcc -arch=sm_35 -ftz=true -I../ -o $EXE $SRC

echo $EXE

K=65536
for i in {1..4}
do
    ./$EXE $K
    K=$(($K*2))
done
