awk -F'\t' 'NR>1 && $6!="" && $7==""{print $6}' /Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/2026-06-29_download/CAZy_characterized/GH43_characterized.tsv > to_map.txt
IDS=$(paste -sd, to_map.txt)

# 1. Submit
JOB=$(curl -s --request POST 'https://rest.uniprot.org/idmapping/run' \
  --form 'from=EMBL-GenBank-DDBJ_CDS' \
  --form 'to=UniProtKB' \
  --form "ids=${IDS}" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["jobId"])')

# 2. Poll
until curl -s "https://rest.uniprot.org/idmapping/status/${JOB}" \
  | grep -q 'FINISHED\|results'; do sleep 2; done

# 3. Results -> TSV (from_id, uniprot_acc)
curl -s "https://rest.uniprot.org/idmapping/uniprotkb/results/${JOB}?format=tsv&fields=accession" \
  > mapped.tsv

cat mapped.tsv