echo korean_learner
python3 run_gleu.py --reference '../../get_data/korean_learner/korean_learner_corrected.txt' --source '../../get_data/korean_learner/korean_learner_original.txt' --hypothesis '../../get_data/korean_learner/korean_learner_original.txt'
echo lang8
python3 run_gleu.py --reference '../../get_data/lang8/lang8_corrected.txt' --source '../../get_data/lang8/lang8_original.txt' --hypothesis '../../get_data/lang8/lang8_original.txt'
echo native
python3 run_gleu.py --reference '../../get_data/native/native_corrected.txt' --source '../../get_data/native/native_original.txt' --hypothesis '../../get_data/native/native_original.txt'
echo union
python3 run_gleu.py --reference '../../get_data/union/union_corrected.txt' --source '../../get_data/union/union_original.txt' --hypothesis '../../get_data/union/union_original.txt'
