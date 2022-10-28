TYPE = SPELL WS PUNCT NA
DIR = RM

wslang8:
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/lang8/lang8_original.txt -cor ../../../extract_data/lang8/lang8_corrected.txt -out ../../../extract_data/lang8/lang8 -save > ws_error/corrected_log/lang8.txt

wsnative:
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/native/native_original.txt -cor ../../../extract_data/native/native_corrected.txt -out ../../../extract_data/native/native -save > ws_error/corrected_log/native.txt
wskl:
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/korean_learner/korean_learner_original.txt -cor ../../../extract_data/korean_learner/korean_learner_corrected.txt -out ../../../extract_data/korean_learner/korean_learner -save > ws_error/corrected_log/korean_learner.txt
ws:
	python3 scripts/extractPos/align.py
debug:
	[ -d alignment_output/$(DIR) ] || mkdir alignment_output/$(DIR)
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/korean_learner_original.txt -cor ../../../sample_data/korean_learner_corrected.txt -out alignment_output/$(DIR)/korean_learner -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/lang8_original.txt -cor ../../../sample_data/lang8_corrected.txt -out alignment_output/$(DIR)/lang8 -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/native_original.txt -cor ../../../sample_data/native_corrected.txt -out alignment_output/$(DIR)/native -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/wiki_original.txt -cor ../../../sample_data/wiki_corrected.txt -out alignment_output/$(DIR)/wiki -save -logset $(TYPE)

pos:
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/korean_learner_original.txt -cor ../../../sample_data/korean_learner_corrected.txt -out alignment_output/$(DIR)/korean_learner -save -logset $(TYPE) > alignment_output/pos/korean_learner.txt
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/lang8_original.txt -cor ../../../sample_data/lang8_corrected.txt -out alignment_output/$(DIR)/lang8 -save -logset $(TYPE) > alignment_output/pos/lang8.txt
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/native_original.txt -cor ../../../sample_data/native_corrected.txt -out alignment_output/$(DIR)/native -save -logset $(TYPE) > alignment_output/pos/native.txt
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/wiki_original.txt -cor ../../../sample_data/wiki_corrected.txt -out alignment_output/$(DIR)/wiki -save -logset $(TYPE) > alignment_output/pos/wiki.txt

sample:
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/korean_learner_original.txt -cor ../../../sample_data/korean_learner_corrected.txt -out alignment_output/$(DIR)/korean_learner -save
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/lang8_original.txt -cor ../../../sample_data/lang8_corrected.txt -out alignment_output/$(DIR)/lang8 -save
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/native_original.txt -cor ../../../sample_data/native_corrected.txt -out alignment_output/$(DIR)/native -save
	python3 parallel_to_m2_korean.py -orig ../../../sample_data/wiki_original.txt -cor ../../../sample_data/wiki_corrected.txt -out alignment_output/$(DIR)/wiki -save

kortest:
	python3 parallel_to_m2_korean.py -orig test/korean/orig -cor test/korean/cor


full:
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/korean_learner/korean_learner_original.txt -cor ../../../extract_data/korean_learner/korean_learner_corrected.txt -out newm2/korean_learner -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/native/native_original.txt -cor ../../../extract_data/native/native_corrected.txt -out newm2/native -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/lang8/lang8_original.txt -cor ../../../extract_data/lang8/lang8_corrected.txt -out newm2/lang8 -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki1_original.txt -cor ../../../extract_data/wiki/wiki1_corrected.txt -out newm2/wiki1 -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki2_original.txt -cor ../../../extract_data/wiki/wiki2_corrected.txt -out newm2/wiki2 -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki3_original.txt -cor ../../../extract_data/wiki/wiki3_corrected.txt -out newm2/wiki3 -save
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki4_original.txt -cor ../../../extract_data/wiki/wiki4_corrected.txt -out newm2/wiki4 -save

debugfull:
	[ -d alignment_output/$(DIR) ] || mkdir alignment_output/$(DIR)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/korean_learner/korean_learner_original.txt -cor ../../../extract_data/korean_learner/korean_learner_corrected.txt -out alignment_output/$(DIR)/korean_learner_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/lang8/lang8_original.txt -cor ../../../extract_data/lang8/lang8_corrected.txt -out alignment_output/$(DIR)/lang8_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/native/native_original.txt -cor ../../../extract_data/native/native_corrected.txt -out alignment_output/$(DIR)/native_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki1_original.txt -cor ../../../extract_data/wiki/wiki1_corrected.txt -out alignment_output/$(DIR)/wiki1_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki2_original.txt -cor ../../../extract_data/wiki/wiki2_corrected.txt -out alignment_output/$(DIR)/wiki2_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki3_original.txt -cor ../../../extract_data/wiki/wiki3_corrected.txt -out alignment_output/$(DIR)/wiki3_full -save -logset $(TYPE)
	python3 parallel_to_m2_korean.py -orig ../../../extract_data/wiki/wiki4_original.txt -cor ../../../extract_data/wiki/wiki4_corrected.txt -out alignment_output/$(DIR)/wiki4_full -save -logset $(TYPE)

sample_m2:
	cat newm2/unclassified_merged/native.m2 > alignment_output/mother_tongue.m2
	cat newm2/unclassified_merged/korean_learner.m2 newm2/unclassified_merged/lang8.m2 > alignment_output/second_language.m2
	cat alignment_output/mother_tongue.m2 alignment_output/second_language.m2 > alignment_output/all_dataset.m2
	python3 rand_sample.py mother_tongue
	python3 rand_sample.py second_language
	python3 rand_sample.py all_dataset 30
