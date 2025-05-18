for i in dog_gpu dog_parallel dog_optimized dog
do
	echo "Running ./$i images/$1 outputs/${i}_$2 $3 $4 $5 $6 $7 $8 $9"
	./$i images/$1 outputs/${i}_$2 $3 $4 $5 $6 $7 $8 $9
done