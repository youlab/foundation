gpu:
	srun -p youlab-gpu --mem=32000 --mincpu=8 --gres=gpu:1 --account=youlab --pty bash -i

shell:
	apptainer shell --nv --bind /hpc/group/youlab/zz294/VAE/saved_sims --bind /hpc/group/youlab/zah8/foundations/ --bind /cwork/zah8/ --bind /work/zah8/ /hpc/group/youlab/you-lab-deep-learning.sif

rm-slurm:
	rm slurm_outputs/*

data:
	python src/run_data.py

ml:
	python src/run_ml.py

applications:
	python src/run_applications.py

comparison:
	python src/run_comparison.py

figs:
	python src/make_figs.py
