#!/bin/sh
SCRIPTS=~/SMT/mosesdecoder/scripts

DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
DETRUECASE=$SCRIPTS/recaser/detruecase.perl


#while true

#do


#!/bin/sh
SCRIPTS=~/SMT/mosesdecoder/scripts

DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
DETRUECASE=$SCRIPTS/recaser/detruecase.perl



prefix="model_wmt14-en-de_tok_joint_bpe32_len256_4096_8_8000_0.98_2/wmt14-en-de_tok_joint_bpe32_len_256_4096_8_8000_0.98_2_model.pt.best_bleu.pt"

if [ -f "$prefix" ]; then

    echo "TEST"

    sleep 5

    # GPU=`python GPU.py`

    # echo $GPU

    five=26

    #python translate.py -gpu 0 -model $prefix -src ~/NMT/pytorch/Data/wmt14/test.tok.bpe32.en -tgt  ~/NMT/pytorch/Data/wmt14/test.tok.bpe32.de -output $prefix.test.pred

    # ## get BLEU
    sed 's/\@\@ //g' < $prefix.test.pred > $prefix.test.true
    #perl $DETRUECASE < $prefix.true > $prefix.detrue
    #perl $DETOKENIZER -l en < $prefix.detrue > $prefix.detok
    #python2 ~/NMT/pytorch/Data/ldc_1.25/plain2sgm-dev.py $prefix.detok ~/NMT/pytorch/Data/ldc_1.25/nist02_src.sgm $prefix.detok.sgm
    #BLEU=`~/NMT/pytorch/Data/ldc_1.25/mteval-v11b-dev.pl -b -r ~/NMT/pytorch/Data/ldc_1.25/nist02_ref.sgm -s ~/NMT/pytorch/Data/ldc_1.25/nist02_src.sgm  -t $prefix.detok.sgm | cut -f 4 -d ' '`
    BLEU=`sacrebleu ~/NMT/pytorch/Data/wmt14/test.tok.de < $prefix.test.true --force `
    #BLEU=`./multi-bleu.perl ~/NMT/pytorch/Data/wmt14/test.tok.de < $prefix.test.true | cut -f 1 -d ',' | cut -f 3 -d ' '`
    #-b


    echo "BLEU = $BLEU"

fi
