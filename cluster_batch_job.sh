#####################################
#!/bin/sh
#$ -binding linear:3
#$ -q cognition-all.q
#$ -cwd
#$ -V
#$ -N job_name
#$ -e temp.err
#$ -o temp.txt
#$ -t 1-5
#####################################

echo "------------------------------------------------------------------------"
. ~/.bashrc && conda activate experiment-env
echo "Successfully activated virtual environment - Ready to start job"
echo "------------------------------------------------------------------------"
echo "Job started on" `date`
echo "------------------------------------------------------------------------"
python run_meta_a3c.py --seed_id $SGE_TASK_ID
echo "------------------------------------------------------------------------"
echo "Job ended on" `date`
echo "------------------------------------------------------------------------"
