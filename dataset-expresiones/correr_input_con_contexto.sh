#!/bin/bash
while read -r word1 word2 context1 context2; do
    python -c "from cosine_similarity import cos_sim; cos_sim($word1, $word2, $context1, $context2)" >> output_con_contexto.txt
done < inputs_con_contexto.txt

