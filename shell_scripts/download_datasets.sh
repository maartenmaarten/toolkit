organism="Bifidobacterium longum infantis"
outdir="b_longum_infantis"

time datasets download genome taxon "$organism" \
    --include genome \
    --assembly-level complete \
    --filename "${outdir}.zip" \
    --dehydrated \
&& unzip "${outdir}.zip" -d "$outdir"

# retry 3 times
time datasets rehydrate --directory "$outdir" 
time datasets rehydrate --directory "$outdir"
time datasets rehydrate --directory "$outdir"
