#!/bin/bash
while read -r word1 word2 ; do
    python -c "from cosine_similarity import cos_sim; cos_sim($word1, $word2)" >> output_sin_contexto.txt
done < inputs_sin_contexto.txt

