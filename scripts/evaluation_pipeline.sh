_this_script=`readlink -f $0`
_this_dir=`dirname $_this_script`

python_env="/usr/prog/sb/mw/python/conda/sb-aegis/bin/python"
qsub_flags="-pe smp 1 -l gpu_card=1 -l gpu_arch=tesla_v100* -l h_rt=12000 -l m_mem_free=64G -cwd -j y -b y "

echo cd4.py

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

echo maria.py

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

echo nod.py

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_295_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth iedb 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_300_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_143_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_145_best_recall.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_294_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod

echo /usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth iedb_nod 

$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_292_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

