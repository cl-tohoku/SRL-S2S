#!/usr/bin/env bash


### INPUT ###
echo "save dir"
echo "write 'ls' if you want to check the dir in 'saved_models'"
read SAVE
if [ ${SAVE} == "ls" ]; then
  ls saved_models/
  read SAVE
fi
SAVE_DIR="saved_models/${SAVE}"

if [ ! -d ${SAVE_DIR} ]; then
  mkdir ${SAVE_DIR}
  echo "make directory: ${SAVE_DIR}"
else
  echo "directory is existed: ${SAVE_DIR}"
fi

if [ ! -d datasets/json ]; then
  mkdir datasets/json
fi

### CoNLL_to_JSON ###
echo "### CoNLL_to_JSON"
RAW_DIR="datasets/conll05"
for data in "train" "dev" "test.wsj" "test.brown"
do
  RAW="${RAW_DIR}/conll05.${data}.txt"
  OUT="datasets/json/conll05.${data}.json"
  if [ ! -f ${OUT} ]; then
    python pre_processing/CoNLL_to_JSON.py \
      --source_file ${RAW} \
      --output_file ${OUT} \
      --dataset_type mono \
      --src_lang "<EN>" \
      --token_type CoNLL05
  else
    echo "existed: ${OUT}"
  fi
done


### TRAIN ###
echo "### TRAIN"
if [ ! -f ${SAVE_DIR}/model.tar.gz ]; then
  echo "config file: training_config/monolingual"
  ls training_config/monolingual
  read CONFIG

  allennlp train training_config/monolingual/${CONFIG} \
    -s ${SAVE_DIR} \
    --include-package src

else
  echo "existed: ${SAVE_DIR}/model.tar.gz"  
fi


### PREDICT ###
echo "### PREDICT"
for TEST in "wsj" "brown"
do
  OUT_PRED="${SAVE_DIR}/predicted.test.${TEST}.json"
  echo ${OUT_PRED}
  if [ ! -f ${OUT_PRED} ]; then
    allennlp predict ${SAVE_DIR}/model.tar.gz \
      datasets/json/conll05.test.${TEST}.json \
      --output-file ${OUT_PRED} \
      --include-package src \
      --predictor seq2seq-srl
  else
    echo "existed: ${OUT_PRED}"
  fi
done


### JSON_to_CoNLL ###
echo "### JSON_to_CoNLL"
J2C="${SAVE_DIR}/JSON_to_CoNLL"
if [ ! -d ${J2C} ]; then
  mkdir ${J2C}
fi
for TEST in "wsj" "brown"
do
  for ORACLE in "min" "max"
  do
    PROP_TEST="${SAVE_DIR}/eval.test.${TEST}.${ORACLE}.prop"
    if [ ! -f ${PROP_TEST} ]; then
      python work/test_from_pred-json_to_srl-eval.py \
        --saved_dir ${SAVE_DIR} \
        --test ${TEST} \
        --oracle ${ORACLE} \
        | tee ${J2C}/${TEST}.${ORACLE}.txt
    else
      echo "existed: ${PROP_TEST}"
    fi
  done
done


### perl ###
echo "### PERL"
PERL="${SAVE_DIR}/PERL"
if [ ! -d ${PERL} ]; then
  mkdir ${PERL}
fi
for TEST in "wsj" "brown"
do
  for ORACLE in "min" "max"
  do
    PROP_GOLD="datasets/conll05/conll05.test.${TEST}.prop"
    PROP_TEST="${SAVE_DIR}/eval.test.${TEST}.${ORACLE}.prop"
    perl ~/soft/srlconll-1.1/bin/srl-eval.pl ${PROP_GOLD} ${PROP_TEST} | tee ${PERL}/${TEST}.${ORACLE}.txt
  done
done

