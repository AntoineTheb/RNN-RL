POLICY=$1

for ((i=0;i<3;i+=1))
do
  python main.py --policy=$POLICY --recurrent --save_model --seed=$i
done


