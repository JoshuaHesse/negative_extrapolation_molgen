SHELL := /bin/bash

DEVELOPMENT_SEED ?= 11
SCOPE_DEVELOPMENT_SEEDS ?= 5 7 11
PAPER_SEEDS ?= 13 17 19 23 29 31 37 41 43 47
FCD_JOBS ?= 8
ANALYSIS_CPUS ?= 64
ANALYSIS_CPUSET ?= 0-63
SEMLAFLOW_CKPT ?= external/semla-flow/assets/models/geom-drugs/200epochs.ckpt
SEMLAFLOW_DATA ?= external/semla-flow/assets/data/geom-drugs/smol
SEMLAFLOW_POSEBUSTERS_WORKERS ?= 16
SEMLAFLOW_POSEBUSTERS_CPUS ?= 64
SEMLAFLOW_POSEBUSTERS_CPUSET ?= 0-63
SEMLAFLOW_FOUR_LIABILITY_ROOT ?= results/external/semlaflow/four_liability_joint_replicates
SEMLAFLOW_FOUR_LIABILITY_ANALYSIS_SAMPLES ?= $(SEMLAFLOW_FOUR_LIABILITY_ROOT)/analysis/smiles_samples_confirmatory
SEMLAFLOW_FOUR_LIABILITY_MODELS ?= base random_tuned positive_tuned bad_tuned full_model_neon_lambda_2p5 full_model_random_neon_lambda_2p5 full_model_norm_matched_random_corrected_neon_lambda_2p5 full_model_positive_corrected_neon_lambda_2p5 full_model_neon_lambda_4 full_model_random_neon_lambda_4 full_model_norm_matched_random_corrected_neon_lambda_4 full_model_positive_corrected_neon_lambda_4

.PHONY: build test gpu-shell guacamol-download guacamol-epoch-sensitivity-scaffolds guacamol-profile guacamol-rnn-assay-interference-removal-from-base guacamol-rnn-charged-motif-removal-from-base guacamol-rnn-chelator-removal-from-base guacamol-rnn-qed-from-base guacamol-rnn-reactive-epoch-sensitivity guacamol-rnn-reactive-removal-from-base guacamol-rnn-specific-liability-objectives guacamol-train-rnn-base guacamol-train-transformer-base guacamol-transformer-assay-interference-lastblock-final-from-base guacamol-transformer-charged-motif-lastblock-final-from-base guacamol-transformer-chelator-lastblock-final-from-base guacamol-transformer-lastblock-specific-liability-objectives guacamol-transformer-qed-from-base guacamol-transformer-reactive-epoch-sensitivity guacamol-transformer-reactive-lastblock-final-from-base paper-control-refresh paper-fcd-distances paper-fcd-mw-calibration paper-fcd-mw-calibration-guacamol-rnn paper-fcd-mw-calibration-guacamol-transformer paper-fcd-mw-calibration-reinvent paper-guacamol-base-replicates paper-guacamol-chelator-diversity paper-guacamol-liability-replicates paper-guacamol-qed-replicates paper-guacamol-random-controls paper-guacamol-rnn-chelator-fcd paper-guacamol-rnn-chelator-reference paper-guacamol-transformer-chelator-fcd paper-post-control-analysis paper-refresh-fdd paper-refresh-guacamol-rnn-fdd paper-refresh-guacamol-transformer-fdd paper-refresh-reinvent-fdd paper-reinvent-chelator-fcd paper-reinvent-liability-replicates paper-reproduction paper-semlaflow-confirmatory paper-si-tables paper-statistics paper-transformer-scope-development-data publication-si-tables reinvent-assay-interference-replicates reinvent-assay-interference-replicates-analyze reinvent-assay-interference-replicates-resume reinvent-build reinvent-charged-motif-replicates reinvent-charged-motif-replicates-analyze reinvent-charged-motif-replicates-resume reinvent-chelator-replicates reinvent-chelator-replicates-analyze reinvent-chelator-replicates-resume reinvent-reactive-epoch-sensitivity reinvent-reactive-replicates reinvent-reactive-replicates-analyze reinvent-reactive-replicates-resume reinvent-shell reinvent-specific-liability-replicates semlaflow-build semlaflow-env-check semlaflow-four-liability-analysis-export semlaflow-four-liability-distribution-distance semlaflow-four-liability-diversity semlaflow-four-liability-joint-replicates semlaflow-four-liability-joint-replicates-posebusters semlaflow-four-liability-joint-replicates-summarize semlaflow-four-liability-paper-analysis semlaflow-four-liability-positive-corrected-replicates semlaflow-four-liability-structural-analysis semlaflow-four-liability-usable-scaffolds semlaflow-geom-drugs-50000-baseline-replicates shell

build:
	docker compose build neon-molgen

test:
	docker compose run --rm neon-molgen pytest -q

.PHONY: audit audit-inputs audit-results figures lint zenodo-archive zenodo-archive-dry-run
lint:
	docker compose run --rm neon-molgen ruff check \
		neon_molgen scripts tests notebooks/paper_plot_data_loaders.py

audit:
	docker compose run --rm neon-molgen python scripts/audit_reproducibility.py

audit-inputs:
	docker compose run --rm neon-molgen python scripts/audit_reproducibility.py --inputs

audit-results:
	docker compose run --rm neon-molgen python scripts/audit_reproducibility.py --results

figures:
	docker compose run --rm neon-molgen jupyter nbconvert \
		--to notebook --execute --ExecutePreprocessor.timeout=-1 \
		--output-dir /tmp --output main_paper_figures.executed.ipynb \
		notebooks/260612_main_paper_figures.ipynb
	docker compose run --rm neon-molgen jupyter nbconvert \
		--to notebook --execute --ExecutePreprocessor.timeout=-1 \
		--output-dir /tmp --output si_figures_and_statistics.executed.ipynb \
		notebooks/260617_si_figures_and_statistics.ipynb

zenodo-archive-dry-run:
	python3 scripts/build_zenodo_archive.py --dry-run

zenodo-archive:
	python3 scripts/build_zenodo_archive.py

semlaflow-build:
	docker compose build semlaflow

semlaflow-env-check:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
		python -c "import torch, lightning, posebusters, semlaflow, neon_molgen; from neon_molgen.scoring import score_smiles; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('lightning ok'); print('semlaflow ok'); print('posebusters ok')"

semlaflow-geom-drugs-50000-baseline-replicates:
	@for seed in $(PAPER_SEEDS); do \
		out_dir=results/external/semlaflow/geom_drugs_50000_seed_$${seed}; \
		if [ -f "$${out_dir}/summary.csv" ]; then \
			echo "Skipping SemlaFlow 50k baseline seed $${seed}: $${out_dir}/summary.csv exists"; \
		else \
			echo "Generating SemlaFlow 50k baseline seed $${seed} -> $${out_dir}"; \
			docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
				python scripts/run_semlaflow_probe.py \
				--ckpt-path $(SEMLAFLOW_CKPT) \
				--data-path $(SEMLAFLOW_DATA) \
				--dataset geom-drugs \
				--dataset-split test \
				--output-dir "$${out_dir}" \
				--save-prefix semlaflow_geom_drugs \
				--n-mols 50000 \
				--batch-cost 4096 \
				--seed "$${seed}" \
				--skip-posebusters; \
		fi; \
	done

semlaflow-four-liability-joint-replicates:
	@set -e; for seed in $(PAPER_SEEDS); do \
		baseline_dir=results/external/semlaflow/geom_drugs_50000_seed_$${seed}; \
		seed_root=results/external/semlaflow/four_liability_joint_replicates/seed_$${seed}; \
		random_dir=$${seed_root}/random_reference; \
		positive_dir=$${seed_root}/positive_tuning; \
		joint_dir=$${seed_root}/joint; \
		score_cache=$${seed_root}/shared/baseline_objective_scores.csv; \
		if [ -f "$${joint_dir}/summary.csv" ]; then \
			echo "Skipping completed four-liability replicate seed $${seed}"; \
			continue; \
		fi; \
		if [ ! -f "$${baseline_dir}/summary.csv" ]; then \
			echo "Missing baseline for seed $${seed}: $${baseline_dir}/summary.csv"; \
			exit 1; \
		fi; \
		echo "[$${seed}] matched random fine-tuning reference"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${random_dir}" \
			--objective-column four_liability_hit \
			--selection-mode random \
			--trained-model-name random_tuned \
			--seed "$${seed}" \
			--train-size 12000 \
			--val-size 1200 \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--eval-samples 5000 \
			--bad-model-only \
			--resume \
			--skip-posebusters \
			--device cuda; \
		echo "[$${seed}] matched liability-free positive fine-tuning"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${positive_dir}" \
			--objective-column four_liability_hit \
			--selection-mode objective_nonhits \
			--trained-model-name positive_tuned \
			--seed "$${seed}" \
			--train-size 12000 \
			--val-size 1200 \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--eval-samples 5000 \
			--reference-summary "$${random_dir}/summary.csv" \
			--reference-models base \
			--bad-model-only \
			--resume \
			--skip-posebusters \
			--device cuda; \
		echo "[$${seed}] balanced four-liability NE"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${joint_dir}" \
			--objective-column four_liability_hit \
			--selection-mode balanced_objective_hits \
			--balanced-objective-columns reactive_hit chelator_hit charged_motif_hit assay_interference_hit \
			--per-objective-train-size 3000 \
			--per-objective-val-size 300 \
			--train-size 12000 \
			--val-size 1200 \
			--seed "$${seed}" \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--random-correction-checkpoint "$${random_dir}/models/random_tuned.ckpt" \
			--random-correction-scale 1.0 \
			--eval-samples 5000 \
			--lambda-values 2.5 4.0 \
			--scopes full_model \
			--reference-summary "$${positive_dir}/summary.csv" \
			--reference-models base positive_tuned \
			--include-trained-model \
			--delete-neon-checkpoints \
			--resume \
			--skip-posebusters \
			--device cuda; \
		echo "[$${seed}] removing transient 446 MB training checkpoints"; \
		rm -f \
			"$${random_dir}/models/random_tuned.ckpt" \
			"$${positive_dir}/models/positive_tuned.ckpt" \
			"$${joint_dir}/models/bad_tuned.ckpt"; \
	done

semlaflow-four-liability-positive-corrected-replicates: semlaflow-four-liability-joint-replicates
	@set -e; for seed in $(PAPER_SEEDS); do \
		baseline_dir=results/external/semlaflow/geom_drugs_50000_seed_$${seed}; \
		seed_root=results/external/semlaflow/four_liability_joint_replicates/seed_$${seed}; \
		random_dir=$${seed_root}/random_reference; \
		positive_dir=$${seed_root}/positive_tuning; \
		joint_dir=$${seed_root}/joint; \
		score_cache=$${seed_root}/shared/baseline_objective_scores.csv; \
		positive_done=$${joint_dir}/samples/full_model_positive_corrected_neon_lambda_4/summary.json; \
		pure_random_done=$${joint_dir}/samples/full_model_random_neon_lambda_4/summary.json; \
		random_control_done=$${joint_dir}/samples/full_model_norm_matched_random_corrected_neon_lambda_4/summary.json; \
		if [ -f "$${positive_done}" ] && [ -f "$${pure_random_done}" ] && [ -f "$${random_control_done}" ]; then \
			echo "Skipping completed positive-corrected NE seed $${seed}"; \
			continue; \
		fi; \
		if [ ! -f "$${baseline_dir}/summary.csv" ]; then \
			echo "Missing baseline for seed $${seed}: $${baseline_dir}/summary.csv"; \
			exit 1; \
		fi; \
		if [ ! -f "$${random_dir}/summary.csv" ]; then \
			echo "Missing completed random-reference summary for seed $${seed}"; \
			exit 1; \
		fi; \
		echo "[$${seed}] reconstructing matched random checkpoint"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${random_dir}" \
			--objective-column four_liability_hit \
			--selection-mode random \
			--trained-model-name random_tuned \
			--seed "$${seed}" \
			--train-size 12000 \
			--val-size 1200 \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--eval-samples 5000 \
			--bad-model-only \
			--resume \
			--skip-posebusters \
			--device cuda; \
		echo "[$${seed}] reconstructing liability-free positive checkpoint"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${positive_dir}" \
			--objective-column four_liability_hit \
			--selection-mode objective_nonhits \
			--trained-model-name positive_tuned \
			--seed "$${seed}" \
			--train-size 12000 \
			--val-size 1200 \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--eval-samples 5000 \
			--reference-summary "$${random_dir}/summary.csv" \
			--reference-models base \
			--bad-model-only \
			--resume \
			--skip-posebusters \
			--device cuda; \
		echo "[$${seed}] positive-corrected four-liability NE"; \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm semlaflow \
			python scripts/run_semlaflow_neon.py \
			--ckpt-path $(SEMLAFLOW_CKPT) \
			--data-path $(SEMLAFLOW_DATA) \
			--baseline-dir "$${baseline_dir}" \
			--baseline-scores-cache "$${score_cache}" \
			--output-dir "$${joint_dir}" \
			--objective-column four_liability_hit \
			--selection-mode balanced_objective_hits \
			--balanced-objective-columns reactive_hit chelator_hit charged_motif_hit assay_interference_hit \
			--per-objective-train-size 3000 \
			--per-objective-val-size 300 \
			--train-size 12000 \
			--val-size 1200 \
			--seed "$${seed}" \
			--bad-epochs 10 \
			--learning-rate 3e-4 \
			--random-correction-checkpoint "$${random_dir}/models/random_tuned.ckpt" \
			--random-correction-scale 1.0 \
			--additional-correction-checkpoint positive="$${positive_dir}/models/positive_tuned.ckpt" \
			--eval-samples 5000 \
			--lambda-values 2.5 4.0 \
			--scopes full_model \
			--reference-summary "$${positive_dir}/summary.csv" \
			--reference-models base positive_tuned \
			--include-trained-model \
			--delete-neon-checkpoints \
			--resume \
			--skip-posebusters \
			--device cuda; \
		rm -f \
			"$${random_dir}/models/random_tuned.ckpt" \
			"$${positive_dir}/models/positive_tuned.ckpt" \
			"$${joint_dir}/models/bad_tuned.ckpt"; \
	done

semlaflow-four-liability-joint-replicates-posebusters:
	@set -e; for seed in $(PAPER_SEEDS); do \
		seed_root=results/external/semlaflow/four_liability_joint_replicates/seed_$${seed}; \
		echo "[$${seed}] PoseBusters"; \
		container_name=semlaflow_posebusters_seed_$${seed}_$$(date +%s); \
		container_id=$$(docker compose run -d --name "$${container_name}" semlaflow \
			python scripts/analyze_semlaflow_posebusters.py \
			--model base=$${seed_root}/random_reference/samples/base \
			--model random_tuned=$${seed_root}/random_reference/samples/random_tuned \
			--model positive_tuned=$${seed_root}/positive_tuning/samples/positive_tuned \
			--model bad_tuned=$${seed_root}/joint/samples/bad_tuned \
			--model standard_ne_2p5=$${seed_root}/joint/samples/full_model_neon_lambda_2p5 \
			--model random_ne_2p5=$${seed_root}/joint/samples/full_model_random_neon_lambda_2p5 \
			--model norm_matched_random_corrected_ne_2p5=$${seed_root}/joint/samples/full_model_norm_matched_random_corrected_neon_lambda_2p5 \
			--model standard_ne_4=$${seed_root}/joint/samples/full_model_neon_lambda_4 \
			--model random_ne_4=$${seed_root}/joint/samples/full_model_random_neon_lambda_4 \
			--model norm_matched_random_corrected_ne_4=$${seed_root}/joint/samples/full_model_norm_matched_random_corrected_neon_lambda_4 \
			--model positive_corrected_ne_2p5=$${seed_root}/joint/samples/full_model_positive_corrected_neon_lambda_2p5 \
			--model positive_corrected_ne_4=$${seed_root}/joint/samples/full_model_positive_corrected_neon_lambda_4 \
			--output-dir $${seed_root}/joint/analysis/posebusters \
			--objective-column four_liability_hit \
			--workers $(SEMLAFLOW_POSEBUSTERS_WORKERS) \
			--resume); \
		docker update --cpus $(SEMLAFLOW_POSEBUSTERS_CPUS) --cpuset-cpus $(SEMLAFLOW_POSEBUSTERS_CPUSET) "$${container_id}" >/dev/null; \
		docker logs -f "$${container_id}" & \
		log_pid=$$!; \
		status=$$(docker wait "$${container_id}"); \
		wait "$${log_pid}" || true; \
		docker rm "$${container_id}" >/dev/null; \
		if [ "$${status}" != "0" ]; then exit "$${status}"; fi; \
	done

semlaflow-four-liability-joint-replicates-summarize:
	docker compose run --rm neon-molgen \
		python scripts/summarize_semlaflow_four_liability_replicates.py \
		--results-root results/external/semlaflow/four_liability_joint_replicates \
		--seeds $(PAPER_SEEDS)

semlaflow-four-liability-analysis-export:
	docker compose run --rm neon-molgen \
		python scripts/export_semlaflow_analysis_samples.py \
		--results-root $(SEMLAFLOW_FOUR_LIABILITY_ROOT) \
		--output-dir $(SEMLAFLOW_FOUR_LIABILITY_ANALYSIS_SAMPLES) \
		--objective-column four_liability_hit \
		--seeds $(PAPER_SEEDS) \
		--models $(SEMLAFLOW_FOUR_LIABILITY_MODELS)

semlaflow-four-liability-diversity: semlaflow-four-liability-analysis-export
	ANALYSIS_CPUS=$(ANALYSIS_CPUS) ANALYSIS_CPUSET=$(ANALYSIS_CPUSET) \
	docker compose -f docker-compose.yml -f docker-compose.analysis.yml run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir $(SEMLAFLOW_FOUR_LIABILITY_ANALYSIS_SAMPLES) \
		--output-dir $(SEMLAFLOW_FOUR_LIABILITY_ROOT)/analysis/diversity \
		--seeds $(PAPER_SEEDS) \
		--models $(SEMLAFLOW_FOUR_LIABILITY_MODELS) \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--cluster-fit-size 5000 \
		--n-clusters 50

semlaflow-four-liability-distribution-distance: semlaflow-four-liability-analysis-export
	ANALYSIS_CPUS=$(ANALYSIS_CPUS) ANALYSIS_CPUSET=$(ANALYSIS_CPUSET) \
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.analysis.yml run --rm \
		-e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir $(SEMLAFLOW_FOUR_LIABILITY_ANALYSIS_SAMPLES) \
		--output-dir $(SEMLAFLOW_FOUR_LIABILITY_ROOT)/analysis/distribution_distance_fcd \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--models $(SEMLAFLOW_FOUR_LIABILITY_MODELS) \
		--reference-smiles liability_free_base=liability_free_base.smi \
		--flush-every 20

semlaflow-four-liability-structural-analysis: semlaflow-four-liability-diversity semlaflow-four-liability-distribution-distance

semlaflow-four-liability-paper-analysis:
	docker compose run --rm neon-molgen \
		python scripts/analyze_semlaflow_four_liability_paper.py \
		--analysis-dir $(SEMLAFLOW_FOUR_LIABILITY_ROOT)/analysis \
		--seeds $(PAPER_SEEDS)

semlaflow-four-liability-usable-scaffolds:
	docker compose run --rm neon-molgen \
		python scripts/analyze_semlaflow_usable_scaffolds.py \
		--results-root $(SEMLAFLOW_FOUR_LIABILITY_ROOT) \
		--seeds $(PAPER_SEEDS)

paper-fcd-mw-calibration: paper-fcd-mw-calibration-reinvent paper-fcd-mw-calibration-guacamol-rnn paper-fcd-mw-calibration-guacamol-transformer

paper-fcd-mw-calibration-reinvent:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_fcd_mw_calibration.py \
		--results-dir results/external/reinvent4/chelator_replicates \
		--output-dir results/external/reinvent4/chelator_replicates/analysis/fcd_mw_calibration \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--flush-every 20

paper-fcd-mw-calibration-guacamol-rnn:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_fcd_mw_calibration.py \
		--results-dir results/objectives/guacamol_rnn_chelator_removal \
		--output-dir results/objectives/guacamol_rnn_chelator_removal/analysis/fcd_mw_calibration \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--flush-every 20

paper-fcd-mw-calibration-guacamol-transformer:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_fcd_mw_calibration.py \
		--results-dir results/objectives/guacamol_transformer_chelator_lastblock_final \
		--output-dir results/objectives/guacamol_transformer_chelator_lastblock_final/analysis/fcd_mw_calibration \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--flush-every 20

paper-refresh-fdd: paper-refresh-guacamol-rnn-fdd paper-refresh-guacamol-transformer-fdd paper-refresh-reinvent-fdd

paper-refresh-guacamol-rnn-fdd:
	docker compose run --rm neon-molgen \
		python scripts/recompute_fdd_columns.py \
		--results-dir results/objectives/guacamol_rnn_chelator_removal \
		--metrics results/objectives/guacamol_rnn_chelator_removal/analysis/distribution_distance_fcd_base_filtered_selected/distribution_distance_metrics.csv \
		--models base positive neon_lambda_1.0 random_neon_lambda_1.0 \
		--reference-smiles base_filtered=base_liability_free_smiles.smi

paper-refresh-guacamol-transformer-fdd:
	docker compose run --rm neon-molgen \
		python scripts/recompute_fdd_columns.py \
		--results-dir results/objectives/guacamol_transformer_chelator_lastblock_final \
		--metrics results/objectives/guacamol_transformer_chelator_lastblock_final/analysis/distribution_distance_fcd_10seed/distribution_distance_metrics.csv \
		--models base positive neon_last_block_output_lambda_1.0 random_neon_last_block_output_lambda_1.0

paper-refresh-reinvent-fdd:
	docker compose run --rm neon-molgen \
		python scripts/recompute_fdd_columns.py \
		--results-dir results/external/reinvent4/chelator_replicates \
		--metrics results/external/reinvent4/chelator_replicates/analysis/distribution_distance_fcd/distribution_distance_metrics.csv \
		--models base positive neon_lambda_1 random_neon_lambda_1

paper-guacamol-rnn-chelator-reference:
	docker compose run --rm neon-molgen \
		python scripts/export_filtered_smiles_reference.py \
		--results-dir results/objectives/guacamol_rnn_chelator_removal \
		--seeds $(PAPER_SEEDS) \
		--input-file base_samples.csv \
		--output-file base_liability_free_smiles.smi \
		--filter-column chelator_hit \
		--filter-value 0

paper-guacamol-rnn-chelator-fcd: paper-guacamol-rnn-chelator-reference
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/objectives/guacamol_rnn_chelator_removal \
		--output-dir results/objectives/guacamol_rnn_chelator_removal/analysis/distribution_distance_fcd_base_filtered_selected \
		--seeds $(PAPER_SEEDS) \
		--sample-size 3000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--models base positive neon_lambda_1.0 random_neon_lambda_1.0 \
		--reference-smiles base_filtered=base_liability_free_smiles.smi \
		--resume \
		--flush-every 20

paper-guacamol-transformer-chelator-fcd:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/objectives/guacamol_transformer_chelator_lastblock_final \
		--output-dir results/objectives/guacamol_transformer_chelator_lastblock_final/analysis/distribution_distance_fcd_10seed \
		--seeds $(PAPER_SEEDS) \
		--sample-size 3000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--models base positive neon_last_block_output_lambda_1.0 random_neon_last_block_output_lambda_1.0 \
		--flush-every 20

paper-reinvent-chelator-fcd:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/external/reinvent4/chelator_replicates \
		--output-dir results/external/reinvent4/chelator_replicates/analysis/distribution_distance_fcd \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cuda:0 \
		--jobs $(FCD_JOBS) \
		--models base positive neon_lambda_1 random_neon_lambda_1 \
		--flush-every 20

paper-statistics:
	docker compose run --rm neon-molgen \
		python scripts/analyze_paper_statistics.py

publication-si-tables: paper-statistics
	docker compose run --rm neon-molgen \
		python scripts/export_si_tables.py \
			--output-dir results/publication/tables

# Backward-compatible alias used during manuscript development.
paper-si-tables: publication-si-tables

paper-transformer-scope-development-data:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/train_base_models.py \
			--config configs/guacamol_transformer_base.json \
			--device cuda \
			--seeds $(SCOPE_DEVELOPMENT_SEEDS) \
			--skip-existing \
			--skip-global-summary-update
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
			--config configs/objectives/guacamol_transformer_chelator_scope_development_from_base.json \
			--device cuda \
			--seeds $(SCOPE_DEVELOPMENT_SEEDS) \
			--skip-existing

paper-guacamol-base-replicates:
	$(MAKE) guacamol-train-rnn-base
	$(MAKE) guacamol-train-transformer-base

paper-guacamol-qed-replicates:
	$(MAKE) guacamol-rnn-qed-from-base
	$(MAKE) guacamol-transformer-qed-from-base

paper-guacamol-liability-replicates:
	$(MAKE) guacamol-rnn-specific-liability-objectives
	$(MAKE) guacamol-transformer-lastblock-specific-liability-objectives

paper-guacamol-random-controls:
	@set -e; for config in \
		configs/objectives/guacamol_rnn_qed_from_base.json \
		configs/objectives/guacamol_transformer_qed_from_base.json \
		configs/objectives/guacamol_rnn_reactive_removal_from_base.json \
		configs/objectives/guacamol_rnn_chelator_removal_from_base.json \
		configs/objectives/guacamol_rnn_charged_motif_removal_from_base.json \
		configs/objectives/guacamol_rnn_assay_interference_removal_from_base.json \
		configs/objectives/guacamol_transformer_reactive_lastblock_final_from_base.json \
		configs/objectives/guacamol_transformer_chelator_lastblock_final_from_base.json \
		configs/objectives/guacamol_transformer_charged_motif_lastblock_final_from_base.json \
		configs/objectives/guacamol_transformer_assay_interference_lastblock_final_from_base.json; do \
		docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
			python scripts/run_objective_from_base.py \
				--config "$${config}" \
				--device cuda \
				--refresh-random-controls \
				--delete-refreshed-checkpoints \
				--skip-existing; \
	done

paper-guacamol-chelator-diversity:
	ANALYSIS_CPUS=$(ANALYSIS_CPUS) ANALYSIS_CPUSET=$(ANALYSIS_CPUSET) \
	docker compose -f docker-compose.yml -f docker-compose.analysis.yml run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/objectives/guacamol_rnn_chelator_removal \
		--output-dir results/objectives/guacamol_rnn_chelator_removal/analysis/diversity_selected_10seed \
		--seeds $(PAPER_SEEDS) \
		--models base positive neon_lambda_1.0 random_neon_lambda_1.0 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50
	ANALYSIS_CPUS=$(ANALYSIS_CPUS) ANALYSIS_CPUSET=$(ANALYSIS_CPUSET) \
	docker compose -f docker-compose.yml -f docker-compose.analysis.yml run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/objectives/guacamol_transformer_chelator_lastblock_final \
		--output-dir results/objectives/guacamol_transformer_chelator_lastblock_final/analysis/diversity_selected_10seed \
		--seeds $(PAPER_SEEDS) \
		--models base positive neon_last_block_output_lambda_1.0 random_neon_last_block_output_lambda_1.0 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50

paper-reinvent-liability-replicates:
	$(MAKE) reinvent-reactive-replicates-resume
	$(MAKE) reinvent-chelator-replicates-resume
	$(MAKE) reinvent-charged-motif-replicates-resume
	$(MAKE) reinvent-assay-interference-replicates-resume

paper-semlaflow-confirmatory: semlaflow-geom-drugs-50000-baseline-replicates semlaflow-four-liability-positive-corrected-replicates semlaflow-four-liability-joint-replicates-posebusters semlaflow-four-liability-joint-replicates-summarize

paper-control-refresh: export NEON_DISABLE_MLFLOW ?= 1

paper-control-refresh: paper-guacamol-base-replicates paper-guacamol-qed-replicates paper-guacamol-liability-replicates paper-guacamol-random-controls paper-reinvent-liability-replicates paper-semlaflow-confirmatory

paper-post-control-analysis: paper-guacamol-chelator-diversity paper-guacamol-rnn-chelator-fcd paper-guacamol-transformer-chelator-fcd paper-reinvent-chelator-fcd paper-fcd-mw-calibration semlaflow-four-liability-structural-analysis semlaflow-four-liability-usable-scaffolds semlaflow-four-liability-paper-analysis

shell:
	docker compose run --rm neon-molgen bash

gpu-shell:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen bash

guacamol-download:
	docker compose run --rm neon-molgen \
		python scripts/download_guacamol.py \
		--splits train

guacamol-profile:
	docker compose run --rm neon-molgen \
		python scripts/profile_smiles_dataset.py \
		--input data/guacamol_v1_train.smiles \
		--name guacamol_v1_train \
		--limit 200000 \
		--score-limit 100000

guacamol-train-rnn-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/train_base_models.py \
		--config configs/guacamol_rnn_base.json \
		--device cuda \
		--skip-existing

guacamol-train-transformer-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/train_base_models.py \
		--config configs/guacamol_transformer_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-qed-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_rnn_qed_from_base.json \
		--device cuda \
		--skip-existing

guacamol-transformer-qed-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_transformer_qed_from_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-chelator-removal-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_rnn_chelator_removal_from_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-reactive-removal-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_rnn_reactive_removal_from_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-reactive-epoch-sensitivity:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_model_epoch_sensitivity.py \
		--config configs/objectives/guacamol_rnn_reactive_removal_from_base.json \
		--seed $(DEVELOPMENT_SEED) \
		--epochs 1 2 5 10 \
		--reference-epoch 2 \
		--lambda-value 1.0 \
		--eval-samples 5000 \
		--output-dir results/development/guacamol_rnn_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--device cuda

guacamol-epoch-sensitivity-scaffolds:
	docker compose run --rm neon-molgen \
		python scripts/analyze_epoch_sensitivity_scaffolds.py \
		--results-dir results/development/guacamol_rnn_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--reference-smiles results/guacamol_rnn_base/reference_canonical_smiles.smi \
		--liability-column reactive_hit
	docker compose run --rm neon-molgen \
		python scripts/analyze_epoch_sensitivity_scaffolds.py \
		--results-dir results/development/guacamol_transformer_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--reference-smiles results/guacamol_transformer_base/reference_canonical_smiles.smi \
		--liability-column reactive_hit

guacamol-rnn-charged-motif-removal-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_rnn_charged_motif_removal_from_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-assay-interference-removal-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_rnn_assay_interference_removal_from_base.json \
		--device cuda \
		--skip-existing

guacamol-rnn-specific-liability-objectives:
	$(MAKE) guacamol-rnn-chelator-removal-from-base
	$(MAKE) guacamol-rnn-reactive-removal-from-base
	$(MAKE) guacamol-rnn-charged-motif-removal-from-base
	$(MAKE) guacamol-rnn-assay-interference-removal-from-base

guacamol-transformer-chelator-lastblock-final-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_transformer_chelator_lastblock_final_from_base.json \
		--device cuda \
		--skip-existing

guacamol-transformer-reactive-lastblock-final-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_transformer_reactive_lastblock_final_from_base.json \
		--device cuda \
		--skip-existing

guacamol-transformer-reactive-epoch-sensitivity:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_model_epoch_sensitivity.py \
		--config configs/objectives/guacamol_transformer_reactive_lastblock_final_from_base.json \
		--seed $(DEVELOPMENT_SEED) \
		--epochs 1 2 5 10 \
		--reference-epoch 1 \
		--lambda-value 1.0 \
		--scope-name last_block_output \
		--eval-samples 5000 \
		--output-dir results/development/guacamol_transformer_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--device cuda

guacamol-transformer-charged-motif-lastblock-final-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_transformer_charged_motif_lastblock_final_from_base.json \
		--device cuda \
		--skip-existing

guacamol-transformer-assay-interference-lastblock-final-from-base:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm neon-molgen \
		python scripts/run_objective_from_base.py \
		--config configs/objectives/guacamol_transformer_assay_interference_lastblock_final_from_base.json \
		--device cuda \
		--skip-existing

guacamol-transformer-lastblock-specific-liability-objectives:
	$(MAKE) guacamol-transformer-chelator-lastblock-final-from-base
	$(MAKE) guacamol-transformer-reactive-lastblock-final-from-base
	$(MAKE) guacamol-transformer-charged-motif-lastblock-final-from-base
	$(MAKE) guacamol-transformer-assay-interference-lastblock-final-from-base

reinvent-build:
	docker compose build reinvent4

reinvent-shell:
	docker compose run --rm reinvent4 bash

reinvent-reactive-replicates:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_external_model.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/reactive_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-reactive-replicates-analyze

reinvent-reactive-replicates-resume:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_external_model.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/reactive_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--skip-existing \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-reactive-replicates-analyze

reinvent-reactive-replicates-analyze:
	docker compose run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/external/reinvent4/reactive_replicates \
		--output-dir results/external/reinvent4/reactive_replicates/analysis/diversity \
		--seeds $(PAPER_SEEDS) \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50
	docker compose run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/external/reinvent4/reactive_replicates \
		--output-dir results/external/reinvent4/reactive_replicates/analysis/distribution_distance \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cpu \
		--jobs 8 \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1

reinvent-reactive-epoch-sensitivity:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_epoch_sensitivity.py \
		--prior external/REINVENT4/priors/reinvent.prior \
		--source-dir results/external/reinvent4/reactive_replicates/seed_$(DEVELOPMENT_SEED) \
		--output-dir results/development/reinvent_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--seed $(DEVELOPMENT_SEED) \
		--epochs 1 2 5 10 \
		--reference-epoch 2 \
		--lambda-value 1.0 \
		--eval-samples 5000 \
		--batch-size 256 \
		--learning-rate 0.0001 \
		--device cuda:0 \
		--skip-existing
	docker compose run --rm neon-molgen \
		python scripts/analyze_epoch_sensitivity_scaffolds.py \
		--results-dir results/development/reinvent_reactive_epoch_sensitivity_seed_$(DEVELOPMENT_SEED) \
		--liability-column reactive_hit

reinvent-chelator-replicates:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_chelator_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/chelator_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-chelator-replicates-analyze

reinvent-chelator-replicates-resume:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_chelator_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/chelator_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--skip-existing \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-chelator-replicates-analyze

reinvent-chelator-replicates-analyze:
	docker compose run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/external/reinvent4/chelator_replicates \
		--output-dir results/external/reinvent4/chelator_replicates/analysis/diversity \
		--seeds $(PAPER_SEEDS) \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50
	docker compose run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/external/reinvent4/chelator_replicates \
		--output-dir results/external/reinvent4/chelator_replicates/analysis/distribution_distance \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cpu \
		--jobs 8 \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1

reinvent-charged-motif-replicates:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_charged_motif_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/charged_motif_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-charged-motif-replicates-analyze

reinvent-charged-motif-replicates-resume:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_charged_motif_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/charged_motif_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--skip-existing \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-charged-motif-replicates-analyze

reinvent-charged-motif-replicates-analyze:
	docker compose run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/external/reinvent4/charged_motif_replicates \
		--output-dir results/external/reinvent4/charged_motif_replicates/analysis/diversity \
		--seeds $(PAPER_SEEDS) \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50
	docker compose run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/external/reinvent4/charged_motif_replicates \
		--output-dir results/external/reinvent4/charged_motif_replicates/analysis/distribution_distance \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cpu \
		--jobs 8 \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1

reinvent-assay-interference-replicates:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_assay_interference_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/assay_interference_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-assay-interference-replicates-analyze

reinvent-assay-interference-replicates-resume:
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm reinvent4 \
		python scripts/run_reinvent_reactive_replicates.py \
		--config configs/reinvent/reinvent4_assay_interference_removal.json \
		--prior external/REINVENT4/priors/reinvent.prior \
		--output-dir results/external/reinvent4/assay_interference_replicates \
		--seeds $(PAPER_SEEDS) \
		--baseline-samples 50000 \
		--eval-samples 10000 \
		--lambda-values 0.25 0.5 0.75 1.0 \
		--device cuda:0 \
		--skip-existing \
		--delete-random-control-models \
		--skip-analysis
	$(MAKE) reinvent-assay-interference-replicates-analyze

reinvent-assay-interference-replicates-analyze:
	docker compose run --rm neon-molgen \
		python scripts/analyze_diversity.py \
		--results-dir results/external/reinvent4/assay_interference_replicates \
		--output-dir results/external/reinvent4/assay_interference_replicates/analysis/diversity \
		--seeds $(PAPER_SEEDS) \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1 \
		--internal-sample-size 2000 \
		--internal-random-pairs 200000 \
		--sphere-sample-size 5000 \
		--cluster-fit-size 5000 \
		--n-clusters 50
	docker compose run --rm neon-molgen \
		python scripts/analyze_distribution_distance.py \
		--results-dir results/external/reinvent4/assay_interference_replicates \
		--output-dir results/external/reinvent4/assay_interference_replicates/analysis/distribution_distance \
		--seeds $(PAPER_SEEDS) \
		--sample-size 5000 \
		--min-size 500 \
		--device cpu \
		--jobs 8 \
		--models base bad positive neon_lambda_0.25 neon_lambda_0.5 neon_lambda_0.75 neon_lambda_1 random_neon_lambda_0.25 random_neon_lambda_0.5 random_neon_lambda_0.75 random_neon_lambda_1

reinvent-specific-liability-replicates:
	$(MAKE) reinvent-reactive-replicates
	$(MAKE) reinvent-chelator-replicates
	$(MAKE) reinvent-charged-motif-replicates
	$(MAKE) reinvent-assay-interference-replicates

paper-fcd-distances: paper-guacamol-rnn-chelator-fcd paper-guacamol-transformer-chelator-fcd paper-reinvent-chelator-fcd paper-fcd-mw-calibration

paper-reproduction: paper-control-refresh paper-post-control-analysis paper-statistics publication-si-tables
