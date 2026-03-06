#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# run_mrbayes.sh
# Run from your TaxaADD directory:  bash run_mrbayes.sh
# ─────────────────────────────────────────────────────────────────

set -e

FASTA="data/toy_5taxa_1000/seq_gen_toy_5_reordered_tree.fasta"
NEXUS="data/toy_5taxa_1000/toy_5.nex"

# ── Step 1: Convert FASTA → NEXUS ────────────────────────────────
echo "Converting FASTA to NEXUS..."
python3 - << 'PYEOF'
from Bio import SeqIO

fasta     = "data/toy_5taxa_1000/seq_gen_toy_5_reordered_tree.fasta"
nexus_out = "data/toy_5taxa_1000/toy_5.nex"

records = list(SeqIO.parse(fasta, "fasta"))
n_taxa  = len(records)
n_sites = len(records[0].seq)

with open(nexus_out, "w") as f:
    f.write("#NEXUS\n\n")
    f.write("BEGIN DATA;\n")
    f.write(f"\tDIMENSIONS NTAX={n_taxa} NCHAR={n_sites};\n")
    f.write("\tFORMAT DATATYPE=DNA MISSING=? GAP=-;\n")
    f.write("\tMATRIX\n")
    for rec in records:
        f.write(f"\t{rec.id:<20} {str(rec.seq)}\n")
    f.write("\t;\n")
    f.write("END;\n\n")
    f.write("BEGIN MRBAYES;\n")
    f.write("\tset autoclose=yes nowarn=yes;\n")
    f.write("\tlset nst=1 rates=equal;\n")
    f.write("\tprset brlenspr=unconstrained:exp(10.0) statefreqpr=fixed(equal);\n")
    f.write("\tmcmcp ngen=500000 samplefreq=500 burninfrac=0.25 nchains=4 printfreq=10000 diagnfreq=5000;\n")
    f.write("\tmcmc;\n")
    f.write("\tsumt burnin=250;\n")
    f.write("\tsump burnin=250;\n")
    f.write("\tquit;\n")
    f.write("END;\n")

print(f"Written: {nexus_out}  ({n_taxa} taxa, {n_sites} sites)")
PYEOF

# ── Step 2: Run MrBayes ──────────────────────────────────────────
echo "Starting MrBayes..."
cd data
nohup mb toy_5taxa_1000/toy_5.nex > toy_5taxa_1000/toy_5_mb.log 2>&1 &
MB_PID=$!
echo "MrBayes running — PID: $MB_PID"
echo "Monitor with:  tail -f data/toy_5taxa_1000/toy_5_mb.log"
cd ..

# ── Step 3: Wait and report ──────────────────────────────────────
echo "Waiting for MrBayes to finish..."
wait $MB_PID
echo ""
echo "Done! Convergence check (want all PSRF close to 1.00):"
grep "PSRF" data/toy_5taxa_1000/toy_5_mb.log | tail -20
echo ""
echo "Run comparison with:"
echo "  python3 sampled_tree_compare.py \\"
echo "      --ours    results_ca_vi/run_20260225_112708_ntaxa5/sampled_trees.nex \\"
echo "      --mrbayes data/toy_5taxa_1000/toy_5.nex.trprobs \\"
echo "      --mb_trees data/toy_5taxa_1000/toy_5.nex.run1.t \\"
echo "      --out_dir  comparison_toy5_1"