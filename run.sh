
train_data=$2
test_data=$3
question=$1
outputfile=$4


if [[ ${question} == "1" ]]; then
python3 nb.py $train_data $test_data $outputfile
fi

if [[ ${question} == "2" ]]; then
python3 b_class.py $train_data $test_data $outputfile
fi



