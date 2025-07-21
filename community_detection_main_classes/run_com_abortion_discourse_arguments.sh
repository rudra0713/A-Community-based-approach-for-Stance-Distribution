#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --job-name=test_com_abortion_discourse_args
#SBATCH --output=class_community_abortion_discourse_args_p_updated_3.out
#SBATCH --error=error_community_abortion_discourse_args_p_updated_3.out
#SBATCH --account=def-mageed
#SBATCH --mail-user=rrs99@cs.ubc.ca
#SBATCH --mail-type=ALL
module load cuda cudnn

module load StdEnv/2020 python/3.8.10

source /scratch/rrs99/venv/sd_dist_venv_3/bin/activate

python3.8 community_detection_class.py --read_graph_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion_200_articles/discourse_full_reason+abortion_200_articles.p' \
                                      --read_c_a_l_path='/scratch/rrs99/Discourse_parser_pipeline/output_data_reason+abortion_200_articles/claim_article_label.p' \
                                      --topic_word='abortion' \
                                      --use_stance_tree=True \
                                      --use_only_arguments=False \
                                      --use_entailment=False \
                                      --use_lm_score=False \
                                      --output_file_name='output_200_articles_discourse_arguments' \
                                      --gold_graph_name='abortion_200_articles_discourse_arguments' \
                                      --predicted_graph_name='abortion_200_articles_discourse_arguments'
