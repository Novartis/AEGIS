_this_script=`readlink -f $0`
_this_dir=`dirname $_this_script`

python_env="/usr/prog/sb/mw/python/conda/sb-aegis/bin/python"
qsub_flags="-pe smp 1 -l gpu_card=1 -l gpu_arch=tesla_v100* -l h_rt=12000 -l m_mem_free=64G -cwd -j y -b y "

export CUDA_VISIBLE_DEVICES=$(/usr/prog/sb/sw/alphafold/helpers/get_gpu_uuid.py)

echo "CD4"
echo "seq_only_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_127_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb
echo "seq_mhc_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_78_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb
echo "seq_only_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_102_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod
echo "seq_mhc_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_83_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod
echo "seq_only_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_129_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod
echo "seq_mhc_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/cd4.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_80_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

echo "MARIA"
echo "seq_only_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_127_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb
echo "seq_mhc_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_78_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb
echo "seq_only_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_102_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod
echo "seq_mhc_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_83_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod
echo "seq_only_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_129_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod
echo "seq_mhc_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/maria.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_80_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

echo "NOD"
echo "seq_only_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb/checkpoints/checkpoint_epoch_127_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb
echo "seq_mhc_source_iedb"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb/checkpoints/checkpoint_epoch_78_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb
echo "seq_only_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_nod/checkpoints/checkpoint_epoch_102_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_nod
echo "seq_mhc_source_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_nod/checkpoints/checkpoint_epoch_83_best_matthews.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_nod
echo "seq_only_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_wo_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_only_source_iedb_nod/checkpoints/checkpoint_epoch_129_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_only_iedb_nod
echo "seq_mhc_source_iedb_nod"
$python_env $_this_dir/../src/mhciipresentation/evaluation/nod.py --model_with_pseudo_path '/usr/prog/sb/sw/AEGIS/AEGIS/logs/features_seq_mhc_source_iedb_nod/checkpoints/checkpoint_epoch_80_best_precision.pth' --results /usr/prog/sb/sw/AEGIS/AEGIS/logs/evaluation/seq_mhc_iedb_nod

