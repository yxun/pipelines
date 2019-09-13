
export AIRFLOW_HOME=$PWD/../

# Init the database
airflow initdb

# Check Python exception
python $AIRFLOW_HOME/dags/tutorial.py

# Metadata Validation
## print the list of active DAGs
airflow list_dags

## print the list of tasks the "tutorial" dag_id
airflow list_tasks tutorial

## print the hierarchy of tasks in the tutorial DAG
airflow list_tasks tutorial --tree

# command layout: command subcommand dag_id task_id date

# testing print_date
airflow test tutorial print_date 2015-06-01

# testing sleep
airflow test tutorial sleep 2015-06-01

# testing templated
airflow test tutorial templated 2015-06-01

# optional start a web server in debug mode in the background
# airflow webserver --debug -p 8081 &
# airflow scheduler &

# start your backfill on a date range
airflow backfill tutorial -s 2015-06-01 -e 2015-06-07

