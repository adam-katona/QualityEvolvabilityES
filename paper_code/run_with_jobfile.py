
import sys
import numpy as np
import math
import copy
import random
import time

import os
import signal
import datetime
import subprocess
import uuid

import json
from pydoc import locate
import shutil
from datetime import datetime


from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster




# This is what happens when we start an experiment:
# - we create a bunch of EXPERIMENT_NAME, config pairs.
# - for each pair we crate a directory, and dump config.json inside
# - for each pair we create and submit a jobfile, wich calls run_with_jobfile.py, passing it the experiment directory as argument
# - when the jobfile is scheduled, it goes to the experiment directory, loads the config and creates a dask_jobqueue cluster, which will submit new jobs for the workers.

# This method of submitting jobs which in turn submit jobs to allocate workers is a bit wasteful, because we need to wait the queue 2 times, but it is very convinient,
#  because the dask_jobqueue does most of the work.



def run_scheduler_and_create_cluster(config,job_folder_path,continue_job):

    # sometimes dask runs into the problem of too many open file descriptors
    # make sure to set the limit to the maximum
    import resource
    soft_limit,hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print("Open file descriptor limit: ",soft_limit," ",hard_limit)
    if soft_limit < hard_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE,(hard_limit,hard_limit)) 
        print("Raised the limit to: ",resource.getrlimit(resource.RLIMIT_NOFILE))


    ###############################
    # SET UP SETTINGS FOR CLUSTER #
    ###############################

    NUM_WORKERS = 67  # max worker per user is 500, with 80 we can run 6 instances concurrently   
    NUM_REPEAT_EXPERIMENT = 1   # repeat the experiment this many times, to see the variance in results

    queue_name = "nodes"
    job_time_limit = '48:00:00'

    USE_WEEK_QUEUE = False  # else it is the normal queue, with 48 hours
    if USE_WEEK_QUEUE is True:
        queue_name = "week"
        job_time_limit = '168:00:00'
        NUM_WORKERS = 50 #30
        # week queue has 400 cpus total
        # maybe aim to use half 200 / 5 = 40 cpu per config 


    PROJECT_PATH = "/users/ak1774/scratch/Evolvability_cheetah/Evolvability-ES"
    # This is needed so the dask workers can import project files
    export_python_path_command = "export PYTHONPATH=${PYTHONPATH}:" + PROJECT_PATH


    ##################
    # CREATE CLUSTER #
    ##################

    if False:  # TEST ########################################
        cluster = LocalCluster(n_workers=4,processes=True,threads_per_worker=1)
        client = Client(cluster)
        config["pop_size"] = 10
        config["batch_size"] = 2
        config["evals_per_step"] = 4
        config["eval_batch_size"] = 2
    else:


        cluster = SLURMCluster(  # these params describe a single worker
            queue= queue_name,  # for viking: nodes or week
            #cores=1,
            #memory='1GB',
            
            cores=1,
            processes = 1,
            memory='4GB',   # was 1GB for Ant, tried 2GB still had warning about not enough memory, 


            walltime= job_time_limit, # '48:00:00',
            project="CS-DCLABS-2019",
            scheduler_options={}, #  {'interface': "bond0"},  # only need to set on the login node, here we are running from a cluster node
            job_extra=["--export=ALL"], # extra #SBATCH options
            env_extra=[export_python_path_command,
                        #"conda activate evo"
                        "source /users/ak1774/scratch/torch_env/bin/activate"]  # extra commands to execute before starting the dask-worker
            )

        print("##### THE JOB SCRIPT: ######")
        print(cluster.job_script())
        print("############################")

        cluster.scale(cores=NUM_WORKERS)
        client = Client(cluster)

    print("Waiting for workers...")
    client.wait_for_workers(1) # wait until there is at least a single worker (it can ramp up later)
    print("First worker registered!", flush=True)


    #####################
    # START EXPERIMENTS #
    #####################

    # If this is a normal job
    if continue_job is None:

        main_experiment_dir = job_folder_path
        for experiment_number in range(NUM_REPEAT_EXPERIMENT):
            
            sub_experiment_dir = os.path.join(main_experiment_dir, str(experiment_number) + datetime.now().strftime("_%m_%d___%H:%M")+"_"+str(np.random.randint(10000000)))
            os.makedirs(sub_experiment_dir,exist_ok=True)
            os.chdir(sub_experiment_dir)

            import main
            main.main(client,**config)
            print("Finished experiment ",experiment_number)

            # go back to the main experiment dir
            os.chdir(main_experiment_dir)

    # this is a job to continue an unfinished job
    else:

        # we are already in the right folder.
        # just need to call main, and tell it that this is a continue job and the details.  

        if "extra_config" not in config:
            config["extra_config"] = {}
        config["extra_config"]["continue_config"] = continue_job

        import main
        main.main(client,**config)
        print("Finished finishing the unfinished experiment :)")




if __name__ == '__main__':

    job_folder_path = sys.argv[1]
    os.chdir(job_folder_path)

    if "continue" in job_folder_path:
        # THis is a continue job
        with open('continue_job.json') as f:
            continue_job = json.load(f)
        
        config_path = continue_job["config_folder"]+"/config.json"
        with open(config_path) as f:
            config = json.load(f)
        run_scheduler_and_create_cluster(config,job_folder_path,continue_job)

    else:
        # This is a normal job
        with open('config.json') as f:
            config = json.load(f)
        run_scheduler_and_create_cluster(config,job_folder_path,continue_job=None)



     