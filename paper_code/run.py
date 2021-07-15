

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

import json
from pydoc import locate
import shutil
from datetime import datetime

import glob


from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster




def get_config_PLAIN_ES_DIRECTIONAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "plain_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "DirectionalAnt"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_PLAIN_ES_NORMAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "plain_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "Ant"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_PLAIN_ES_DECEPTIVE_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "plain_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "DeceptiveAnt" 
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME



def get_config_NONDOMINATED_ES_DIRECTIONAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "nondominated_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "DirectionalAnt"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_NONDOMINATED_ES_NORMAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "nondominated_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "Ant"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_NONDOMINATED_ES_DECEPTIVE_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "nondominated_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "DeceptiveAnt"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME



def get_config_EVO_ES_DIRECTIONAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "maxvar_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "DirectionalAnt"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_EVO_ES_NORMAL_ANT(config):
    config = copy.deepcopy(config)
    config["n_iterations"] = 200
    config["ES_type"] = "maxvar_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = "Ant"  # Ant
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME


def get_all_ant_configs():

    configs = []
    experiment_names = []

    get_conf_funs = [
        get_config_PLAIN_ES_DIRECTIONAL_ANT,
        get_config_PLAIN_ES_NORMAL_ANT,
        get_config_PLAIN_ES_DECEPTIVE_ANT,

        get_config_NONDOMINATED_ES_DIRECTIONAL_ANT,
        get_config_NONDOMINATED_ES_NORMAL_ANT,
        get_config_NONDOMINATED_ES_DECEPTIVE_ANT,

        get_config_EVO_ES_DIRECTIONAL_ANT,
        get_config_EVO_ES_NORMAL_ANT,
    ]

    for fun in get_conf_funs:
        config,EXPERIMENT_NAME = fun(default_config)
        configs.append(config)
        experiment_names.append(EXPERIMENT_NAME)

    return configs,experiment_names

def get_deceptive_ant_configs():

    configs = []
    experiment_names = []

    get_conf_funs = [
        get_config_PLAIN_ES_DECEPTIVE_ANT,
        get_config_NONDOMINATED_ES_DECEPTIVE_ANT,
    ]

    for fun in get_conf_funs:
        config,EXPERIMENT_NAME = fun(default_config)
        configs.append(config)
        experiment_names.append(EXPERIMENT_NAME)

    return configs,experiment_names
    


def get_config_PLAIN_ES(env_name,config):
    config = copy.deepcopy(config)
    config["ES_type"] = "plain_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = env_name  
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_NONDOMINATED_ES(env_name,config):
    config = copy.deepcopy(config)
    config["ES_type"] = "nondominated_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = env_name  
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME

def get_config_EVO_ES(env_name,config):
    config = copy.deepcopy(config)
    config["ES_type"] = "maxvar_ES"  # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
    config["env_name"] = env_name  
    EXPERIMENT_NAME = config["ES_type"] + "_" + config["env_name"]
    return config,EXPERIMENT_NAME


def get_all_humanoid_configs():

    configs = []
    experiment_names = []

    env_name = "humanoid"
    config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    env_name = "directionalhumanoid"
    config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    #config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config) # SKIP THIS,

    env_name = "deceptivehumanoid"
    config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    return configs,experiment_names


def get_some_humanoid_configs():
    configs = []
    experiment_names = []

    env_name = "humanoid"
    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    env_name = "directionalhumanoid"
    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    env_name = "deceptivehumanoid"
    config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
    config["n_iterations"] = 1200
    configs.append(config)
    experiment_names.append(EXPERIMENT_NAME)

    return configs,experiment_names


def generate_job_file(job_folder_path,experiment_name,USE_WEEK_QUEUE):

    lines = []
    lines.append("#!/bin/bash")
    if USE_WEEK_QUEUE is True:
        lines.append("#SBATCH --partition=week")
        lines.append("#SBATCH --time=168:00:00") 
    else:
        # we dont specify a partition, will default to normal queue
        lines.append("#SBATCH --time=48:00:00") 

    lines.append("#SBATCH --job-name=dask_scheduler_" + experiment_name)
    lines.append("#SBATCH --account=CS-DCLABS-2019")
    lines.append("#SBATCH --export=ALL")
    lines.append("#SBATCH --ntasks=1")
    # --cpus-per-task can be used to request multiple cpus on the same node, so multithreaded code works.
    lines.append("#SBATCH --cpus-per-task=4") # The main thread is responsible for calulation the weighted average, few extra cpus are good to have
    lines.append("#SBATCH --mem=16gb")
    
    
    lines.append("")
    lines.append("python /users/ak1774/scratch/Evolvability_cheetah/Evolvability-ES/run_with_jobfile.py " + job_folder_path)

    return lines


def get_last_iteration_of_run(run_folder):
    files = glob.glob(run_folder + "/*")
    files = [file.split("/")[-1] for file in files]
    latest_iteration = max([int(file.replace("niche_","").replace(".pickle","")) for file in files if "niche" in file])
    return latest_iteration

def get_continue_job_details(CONFIG_FOLDERS):

    runs_to_continue = []
    for CONFIG_FOLDER in CONFIG_FOLDERS:
        subfolders = glob.glob(CONFIG_FOLDER + "/*/")
        subfolders = [folder for folder in subfolders if "dask-worker-space" not in folder]
        
        for folder in subfolders:
            last_iter = get_last_iteration_of_run(folder)

            if last_iter < 799:
                runs_to_continue.append({
                    "config_folder" : CONFIG_FOLDER,
                    "run_folder" : folder,
                    "last_iter" : last_iter,
                    "niche_filename" : folder + "niche_"+str(last_iter)+".pickle",
                    "pop_filename" : folder + "pop_"+str(last_iter)+".pickle",            
                })

    return runs_to_continue

def get_all_config_folders():

    root_folders = [
        "LONG_RESUBMIT_01_18___14:54",
        "NODES_01_14___12:19",
        "NODES_01_09___12:04",
        "NODES_12_28___14:50",
        #"TEST_12_23___15:08",  # fitness is fucked up 
        #"NODES_01_14___10:09",   # just started running
        "NODES_12_31___18:46",
        #"WEEK_12_28___14:45",  # some did not run properly because not enough workers
        "WEEK_NONDOMINATED_ONLY_01_14___12:25"
    ]

    CONFIG_TYES = ["plain_ES_humanoid","nondominated_ES_humanoid","maxvar_ES_humanoid",
        "plain_ES_deceptivehumanoid","nondominated_ES_deceptivehumanoid",
        "plain_ES_directionalhumanoid","nondominated_ES_directionalhumanoid"]

    root_folders = ["/users/ak1774/scratch/Evolvability_cheetah/job_results/"+f for f in root_folders]
    all_config_folders = []
    for root_f in root_folders:
        for c in CONFIG_TYES:
            all_config_folders.append(root_f+"/"+c)

    return all_config_folders



def run_continue_jobs():

    # for this job we dont need to specify configs and stuff, because we just want to continue existing jobs.
    
    NUM_PARRALEL_JOBS = 4
    USE_WEEK_QUEUE = False

    CONFIG_FOLDERS = get_all_config_folders()

    continue_jobs = get_continue_job_details(CONFIG_FOLDERS)
    print("Number of jobs to continue: ",len(continue_jobs))

    current_i = 0
    while True:

        done = False
        # very sophisticated method to check how many jobs are running
        cmd = "squeue -u ak1774 | grep sch | wc -l"
        ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        process_output = ps.communicate()[0]
        if len(process_output) > 10:
            # the process output is too large, there were probably some error
            # For example squeue can fail with: slurm_load_jobs error: Socket timed out on send/recv operation
            continue
        
        num_schedulers_running = int(process_output.decode("utf-8"))

        free_slots = NUM_PARRALEL_JOBS - num_schedulers_running
        for _ in range(free_slots):
            
            if current_i < len(continue_jobs):
                print("submitting continue job: ",current_i,"/",len(continue_jobs))
                print(continue_jobs[current_i]["run_folder"],)

                current_continue_job = continue_jobs[current_i]

                run_dir = current_continue_job["run_folder"]
                continue_dir_path = current_continue_job["run_folder"] + "continue"
                os.makedirs(continue_dir_path,exist_ok=True)
                os.chdir(continue_dir_path)

                current_continue_job
                with open("continue_job.json", 'w') as outfile:
                    json.dump(current_continue_job,outfile, indent=4)

                lines = generate_job_file(job_folder_path=continue_dir_path,experiment_name="continue",USE_WEEK_QUEUE=USE_WEEK_QUEUE)

                # save job file
                with open("job.job", 'w') as f:
                    for line in lines:
                        f.write("%s\n" % line)


                # submit job file
                subprocess.run(["sbatch", "job.job"])
                print("JOB SUBMITTED, ",current_i)

                current_i = current_i + 1

            else:
                print("Last job submitted, we are DONE here.")
                done = True

        if done is True:
            break

        time.sleep(30)





if __name__ == '__main__':


    if True:
        run_continue_jobs()
        print("QUIT")
        sys.exit()
    
    print("QUIT FAILED")

    default_config = {
        "ES_type" : "nondominated_ES",   # can be plain_ES,maxvar_ES,maxent_ES,nondominated_ES
        "ModelType" : "plain_FF", #"CPPN", # can be plain_FF,CPPN
        "ModelExtraArgs" : {"use_positional_encoding" : True,
                            "use_bottleneck" : False,
                            "query_list_positional_encoding_max_wavelength" : 2000,
                            "query_list_positional_encoding_width" : 32,
                            "embedding_positional_encoding_max_wavelength" : 2000,
                            "embedding_positional_encoding_width" : 32,
                            "embedding_positional_encoding_resolution" : 500},

        "learning_rate" : 0.01,
        "batch_size" : 10,#10,
        "pop_size" : 10000,#10000,
        "l2_coeff" : 0.005,
        "noise_std" : 0.02,
        "noise_decay" : 1.0,
        "n_iterations" : 1500,  # was 200 for ant, was 800 for humanoid
        "n_rollouts" : 1,
        "env_deterministic" : True,
        "returns_normalization" : "centered_ranks",
        "evals_per_step" : 100, # was 100
        "eval_batch_size" : 10, # was 10
        "single_threaded" : True,
        "gpu_mem_frac" : 0.2,
        "action_noise" : 0.00,
        "progress" : False,
        "seed" : None, # was 42, only Ant was using it, probably only relevant for eval, because on pop we have random noise in parameters.
        "niche_path" : "niche_{iteration}.pickle",
        "pop_path" : "pop_{iteration}.pickle",
        "po_path" : "po_gen_{iteration}.pickle",
        "device" : "cpu",
        "log_fnm" : "log.json",
        "env_name" : "DirectionalAnt",
        "extra_config" : {
            "HUMANOID_USE_BUILT_IN_REWARDS" : False,
        }
        }


    RUN_LOCALLY = False
    
    RUN_CONTINUOUS_SUBMISSION = True
    

    if RUN_CONTINUOUS_SUBMISSION is True:


        # for this mode, we continuously monitor the running jobs, and if there is a spot, we submit a new one.
        # I dont have to resubmit every 48 hours, and there is no half finished run.
        # each job is only running a single run.
        # we go around the configs, so each config will have the same amount of runs.
        
        # To get the number of active jobs, we check the number of schedulaer jobs running.
        # "squeue -u ak1774 | grep "sch" | wc -l"

        NUM_PARRALEL_JOBS = 4
        USE_WEEK_QUEUE = False

        SUBMISSION_NAME = "RESUBMIT_DECEPTIVE_ANT_PLUS"
        RESULTS_ROOT = "/users/ak1774/scratch/Evolvability_cheetah/job_results"
        os.makedirs(RESULTS_ROOT,exist_ok=True)

        SUBMISSION_ROOT = os.path.join(RESULTS_ROOT, SUBMISSION_NAME + datetime.now().strftime("_%m_%d___%H:%M"))
        os.makedirs(SUBMISSION_ROOT,exist_ok=True)

        #configs,experiment_names = get_all_humanoid_configs()
        configs,experiment_names = get_deceptive_ant_configs()

        current_i = 0
        while True:

            cmd = "squeue -u ak1774 | grep sch | wc -l"
            ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            process_output = ps.communicate()[0]
            num_schedulers_running = int(process_output.decode("utf-8"))

            free_slots = NUM_PARRALEL_JOBS - num_schedulers_running
            for _ in range(free_slots):
                
                current_experiment_i = current_i % len(configs)
                current_experiment_name = experiment_names[current_experiment_i]
                current_config = configs[current_experiment_i]
                

                
                main_experiment_dir = os.path.join(SUBMISSION_ROOT, current_experiment_name)
                os.makedirs(main_experiment_dir,exist_ok=True)
                os.chdir(main_experiment_dir)
                # save job config list, dont care if it already exsist
                with open("config.json", 'w') as outfile:
                    json.dump(current_config,outfile, indent=4)
                # generate job file
                lines = generate_job_file(main_experiment_dir,current_experiment_name,USE_WEEK_QUEUE)

                # save job file
                with open("job.job", 'w') as f:
                    for line in lines:
                        f.write("%s\n" % line)

                # submit job file
                subprocess.run(["sbatch", "job.job"])
                print("JOB SUBMITTED, ",current_i," ",current_experiment_name)

                current_i += 1

            time.sleep(30)


    if RUN_LOCALLY == False:

        SUBMISSION_NAME = "NONDOMINATED_ONLY"
        USE_WEEK_QUEUE = False

        RESULTS_ROOT = "/users/ak1774/scratch/Evolvability_cheetah/job_results"
        os.makedirs(RESULTS_ROOT,exist_ok=True)

        SUBMISSION_ROOT = os.path.join(RESULTS_ROOT, SUBMISSION_NAME + datetime.now().strftime("_%m_%d___%H:%M"))
        os.makedirs(SUBMISSION_ROOT,exist_ok=True)

        #################################
        # CREATE A LIST OF CONFIGS
        ##################################

        #configs,experiment_names = get_all_humanoid_configs()
        #configs,experiment_names = get_some_humanoid_configs()  # some runs take longer, only run those so we have enough of those too
        configs,experiment_names = get_deceptive_ant_configs()

        ##############
        # SUBMIT JOBS
        ##############

        for config,EXPERIMENT_NAME in zip(configs,experiment_names):

            main_experiment_dir = os.path.join(SUBMISSION_ROOT, EXPERIMENT_NAME)

            os.makedirs(main_experiment_dir,exist_ok=True)
            os.chdir(main_experiment_dir)

            # save job config list
            with open("config.json", 'w') as outfile:
                json.dump(config,outfile, indent=4)

            # generate job file
            lines = generate_job_file(main_experiment_dir,EXPERIMENT_NAME,USE_WEEK_QUEUE)

            # save job file
            with open("job.job", 'w') as f:
                for line in lines:
                    f.write("%s\n" % line)

            # submit job file
            subprocess.run(["sbatch", "job.job"])
            print("JOB SUBMITTED, "+EXPERIMENT_NAME)

        print("All jobs submitted!!")






    else:


        RESULTS_ROOT = "/users/ak1774/scratch/Evolvability_cheetah/runs_humanoid"

        # Select environment
        ENV = "SLURM"
        #ENV = "LOCAL"
        #ENV = "CSGPU_LOCAL"

        NUM_WORKERS = 60  # max worker per user is 500, with 80 we can run 6 instances concurrently   
        NUM_REPEAT_EXPERIMENT = 500   # repeat the experiment this many times, to see the variance in results

        queue_name = "nodes"
        job_time_limit = '48:00:00'

        USE_WEEK_QUEUE = False  # else it is the normal queue, with 48 hours
        if USE_WEEK_QUEUE is True:
            queue_name = "week"
            job_time_limit = '168:00:00'
            NUM_WORKERS = 40
            # week queue has 400 cpus total
            # maybe aim to use half 200 / 5 = 40 cpu per config 


        DUBUG_SCALE = False
        if DUBUG_SCALE is True:
            NUM_WORKERS = 5
            config["pop_size"] = 10
            config["batch_size"] = 2
            config["evals_per_step"] = 4
            config["eval_batch_size"] = 2


        #################################
        # SELCT 1 CONFIG TO RUN LOCALLY
        ##################################

        #config,EXPERIMENT_NAME = get_config_PLAIN_ES_DIRECTIONAL_ANT(default_config)
        #config,EXPERIMENT_NAME = get_config_PLAIN_ES_NORMAL_ANT(default_config)
        #config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES_DIRECTIONAL_ANT(default_config)
        #config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES_NORMAL_ANT(default_config)
        #config,EXPERIMENT_NAME = get_config_EVO_ES_DIRECTIONAL_ANT(default_config)   # SKIP THIS, since it is almost the same as EVO_ES_NORMAL_ANT
        #config,EXPERIMENT_NAME = get_config_EVO_ES_NORMAL_ANT(default_config)


        env_name = "humanoid"
        #config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
        #config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
        #config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config)

        env_name = "directionalhumanoid"
        #config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
        #config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
        #config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config) # SKIP THIS,

        env_name = "deceptivehumanoid"
        #config,EXPERIMENT_NAME = get_config_PLAIN_ES(env_name,default_config)
        #config,EXPERIMENT_NAME = get_config_NONDOMINATED_ES(env_name,default_config)
        config,EXPERIMENT_NAME = get_config_EVO_ES(env_name,default_config)





        main_experiment_dir = os.path.join(RESULTS_ROOT, EXPERIMENT_NAME)
        os.makedirs(main_experiment_dir,exist_ok=True)
        os.chdir(main_experiment_dir)


        if "SLURM" in ENV:
            PROJECT_PATH = "/users/ak1774/scratch/Evolvability_cheetah/Evolvability-ES"
        elif ENV == "LOCAL":
            PROJECT_PATH = "/users/ak1774/scratch/Evolvability_cheetah/Evolvability-ES"
        elif ENV == "CSGPU_LOCAL":
            PROJECT_PATH = "/home/userfs/a/ak1774/workspace/Evolvability_cheetah/Evolvability-ES"
        else:
            raise "set up project path!!"

        # This is needed so the dask workers can import project files
        export_python_path_command = "export PYTHONPATH=${PYTHONPATH}:" + PROJECT_PATH




        ####################
        ## SET UP CLUSTER ##
        ####################

        # Befor we set up the cluster, increase the number of allowed open files
        import resource
        soft_limit,hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        print("Open file descriptor limit: ",soft_limit," ",hard_limit)
        if soft_limit < hard_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE,(hard_limit,hard_limit))
            print("Raised the limit to: ",resource.getrlimit(resource.RLIMIT_NOFILE))

        if ENV == "SLURM":

            cluster = SLURMCluster(  # these params describe a single worker
            queue= queue_name,  # for viking: nodes or week
            #cores=1,
            #memory='1GB',
            
            cores=1,
            processes = 1,
            memory='4GB',   # was 1GB for Ant, tried 2GB still had warning about not enough memory, 


            walltime= job_time_limit, # '48:00:00',
            project="CS-DCLABS-2019",
            scheduler_options={'interface': "bond0"},  # this is the interface on the login node (the default is not visible to the workers)
            job_extra=["--export=ALL"], # extra #SBATCH options
            env_extra=[export_python_path_command,
                        #"conda activate evo"
                        "source /users/ak1774/scratch/torch_env/bin/activate"]  # extra commands to execute before starting the dask-worker
            )

            print("##### THE JOB SCRIPT: ######")
            print(cluster.job_script())
            print("############################")

            #cluster.scale(jobs=NUM_WORKERS)  # this will submit the jobs, they will register to the sceduler whenever they got executed
            cluster.scale(cores=NUM_WORKERS)
            client = Client(cluster)

        elif ENV == "LOCAL" or ENV == "CSGPU_LOCAL":
            cluster = LocalCluster(n_workers=NUM_WORKERS,processes=True,threads_per_worker=1)
            client = Client(cluster)
        else:
            raise "Choose a valid env"

        


        # dump config 
        with open("config.json", 'w') as outfile:
            json.dump(config,outfile)

        # start evolution
        print("Waiting for workers...")
        client.wait_for_workers(1) # wait until there is at least a single worker (it can ramp up later)
        print("First worker registered!")

        for experiment_number in range(NUM_REPEAT_EXPERIMENT):
            
            sub_experiment_dir = os.path.join(main_experiment_dir, str(experiment_number) + datetime.now().strftime("_%m_%d___%H:%M"))
            os.makedirs(sub_experiment_dir,exist_ok=True)
            os.chdir(sub_experiment_dir)

            import main
            main.main(client,**config)
            print("Finished experiment ",experiment_number)

            # go back to the main experiment dir
            os.chdir(main_experiment_dir)



        