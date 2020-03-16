#!/bin/sh
SCRIPTS=~/SMT/mosesdecoder/scripts

DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
DETRUECASE=$SCRIPTS/recaser/detruecase.perl


while true

do


prefix="model_wmt14-en-de_tok_joint_bpe32_len256_4096_8_8000_0.98_2/wmt14-en-de_tok_joint_bpe32_len_256_4096_8_8000_0.98_2_model.pt"

if [ -f "$prefix" ]; then

    echo "TEST"

    sleep 20

    # GPU=`python GPU.py`

    # echo $GPU

    five=25

    python translate.py -gpu 1 -model $prefix -src ~/NMT/pytorch/Data/wmt14/dev.tok.bpe32.en -tgt  ~/NMT/pytorch/Data/wmt14/dev.tok.bpe32.de -output $prefix.pred -batch_size 10

    ## get BLEU
    BEST=`cat ${prefix}_best_bleu || echo 0`
    sed 's/\@\@ //g' < $prefix.pred > $prefix.true
    
    BLEU=`./multi-bleu.perl ~/NMT/pytorch/Data/wmt14/dev.tok.de < $prefix.true | cut -f 1 -d ',' | cut -f 3 -d ' '`

    echo $BLEU >> ${prefix}_bleus
    BETTER=`echo "$BLEU >= $BEST" | bc`

    echo "BLEU = $BLEU"

    # save model with highest BLEU
    if [ "$BETTER" = "1" ]; then
        echo "new best; saving"
        echo $BLEU > ${prefix}_best_bleu
        cp ${prefix} ${prefix}.best_bleu.pt
        #python mail.py $prefix'_BLEU_'$BLEU_true
        send=`echo "$BLEU > $five" | bc`
        if [ "$send" = "1" ]; then
            python mail.py $prefix'_BLEU_'$BLEU
        fi
    fi

    rm -rf ${prefix}

fi


done

