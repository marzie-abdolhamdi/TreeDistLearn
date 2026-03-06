#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# run_mrbayes_DS1.sh  —  MrBayes on DS1 synthetic 27-taxon dataset
# Run from your TaxaADD directory:  bash run_mrbayes_DS1.sh
# ─────────────────────────────────────────────────────────────────

set -e

FASTA="data/DS1_synth_1000/seq_gen_1000_DS1_reordered.fasta"
NEXUS="data/DS1_synth_1000/seq_gen_DS1_for_mb.nex"
LOG="data/DS1_synth_1000/DS1_seq_gen_mb.log"

# ── Step 1: Convert FASTA → NEXUS ────────────────────────────────
echo "Converting FASTA to NEXUS..."
python3 - << 'PYEOF'
from Bio import SeqIO

fasta     = "data/DS1_synth_1000/seq_gen_1000_DS1_reordered.fasta"
nexus_out = "data/DS1_synth_1000/seq_gen_DS1_for_mb.nex"

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
    # 27 taxa needs more generations to converge than 5 taxa.
    # 2M generations, sample every 1000 = 2000 samples, 25% burnin = 500 discarded.
    f.write("\tmcmcp ngen=2000000 samplefreq=1000 burninfrac=0.25 nchains=4 printfreq=50000 diagnfreq=10000;\n")
    f.write("\tmcmc;\n")
    f.write("\tsumt burnin=500;\n")
    f.write("\tsump burnin=500;\n")
    f.write("\tquit;\n")
    f.write("END;\n")

print(f"Written: {nexus_out}  ({n_taxa} taxa, {n_sites} sites)")
PYEOF

# ── Step 2: Run MrBayes ──────────────────────────────────────────
echo "Starting MrBayes on DS1 (27 taxa)..."
echo "Expected runtime: 10-30 min depending on server load."
cd data/DS1_synth_1000
nohup mb seq_gen_DS1_for_mb.nex > DS1_seq_gen_mb.log 2>&1 &
MB_PID=$!
cd ../..
echo "MrBayes running — PID: $MB_PID"
echo "Monitor with:  tail -f data/DS1_synth_1000/DS1_seq_gen_mb.log"
echo ""
echo "Watch convergence diagnostics live (printed every 10000 generations):"
echo "  grep 'Average standard deviation' data/DS1_synth_1000/DS1_seq_gen_mb.log"
echo "  (want value < 0.01 before run ends)"

# ── Step 3: Wait and report ──────────────────────────────────────
echo ""
echo "Waiting for MrBayes to finish..."
wait $MB_PID
echo ""
echo "Done! Convergence check (want all PSRF close to 1.00):"
grep "PSRF" data/DS1_synth_1000/DS1_seq_gen_mb.log | tail -20
echo ""
echo "Average standard deviation of split frequencies (want < 0.01):"
grep "Average standard deviation" data/DS1_synth_1000/DS1_seq_gen_mb.log | tail -5
echo ""
echo "Output files:"
echo "  data/DS1_synth_1000/DS1_synth.nex.trprobs   ← topology probabilities"
echo "  data/DS1_synth_1000/DS1_synth.nex.run1.t    ← MCMC tree samples"
echo ""
