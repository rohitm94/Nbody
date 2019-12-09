SRC=nbody-unroll.cu
EXE=nbody-unroll
module load cuda
nvcc -arch=sm_35 -ftz=true -I../ -o $EXE $SRC -DSHMOO

echo $EXE

K=65536
for i in {1..4}
do
    ./$EXE $K
    K=$(($K*2))
done

