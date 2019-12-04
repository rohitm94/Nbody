SRC=gpu_opt.cu
EXE=gpu_opt

#nvcc -arch=sm_35 -I../ -DSHMOO -o $EXE $SRC
 
echo $EXE

K=1024
for i in {1..10}
do
    ./$EXE $K
    K=$(($K*2))
done

