for i in {1..8}; do
    python -m research.experiments.experiment$i > research/results/experiment$i.txt
done
