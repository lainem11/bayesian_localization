"""
Runs hyperparameter search for the E-field optimizer.
"""
import torch
import random
import os
import ray
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import multiprocessing
from ABL import LocalizationSession, preprocessing
import jax

seed = 17
wrkdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
EXAMPLE_SESSION = os.path.join(wrkdir, "examples", "saved", "example_subject", "example_session")

print(f"Is CUDA available: {torch.cuda.is_available()}")

def run_search(config):
    from ABL import LocalizationSession, preprocessing
    import torch
    import random
    import os
    import jax
    from ray import tune

    # Set seeds inside the function to ensure each trial is reproducible
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = EXAMPLE_SESSION

    S = LocalizationSession.load_experiment(save_dir=save_dir,silent_mode=True)
    
    # Store trial data and then reset state
    efield_trials = S.get_active_efield_trials()
    responses = S.get_active_responses()
    S.initialize_trials()

    # Load optimizer settings
    for key,value in config.items():
        S.optimizer.optimizer_settings[key] = value
    
    trial_count = 100
    objective_value = 0.0
    
    for i in range(S.trial_number, trial_count):
        S.start_trial()
        S.choose_next_stimulus()
        
        objective_value += float(S.optimizer.losses[-1].item())

        # Replace trial data from import, but maintain trial number
        current_trial_number = S.trial_number
        S.import_trial_data(efield_trials, responses)
        S.data_handler.trial_number = current_trial_number

        S.fit_model_to_data()
        S.update_source_probabilities()
        S.complete_trial()

        tune.report({"goodness": objective_value})

    # Clean up
    S.end_session()
    del S

if __name__ == "__main__":

    n_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count()
    else:
        n_cpus = int(n_cpus)
    print(f"Using {n_cpus} CPU cores.\n")

    n_gpus = os.getenv('SLURM_GPUS')
    if n_gpus is None:
        n_gpus = 1
    else:
        n_gpus = int(n_gpus)

    print(f"SLURM allocated {n_cpus} CPUs and {n_gpus} GPUs.")

    ray.init(num_cpus=n_cpus, num_gpus=n_gpus)

    print("--- Ray Status ---")
    os.system("ray status")
    print("--------------------")
 
    # Resource Allocation per Trial
    cpus_per_trial = 1
    if n_gpus > 0:
        gpus_per_trial = 1 / (n_cpus / n_gpus) if n_cpus > n_gpus else 1
        max_concurrent_trials = int(n_gpus / gpus_per_trial)
    else:
        gpus_per_trial = 0
        max_concurrent_trials = max(1, int(n_cpus / cpus_per_trial))

    # Hyperparameter Search Space
    search_space = {
        'popsize': tune.choice([128,256,512]),
        'crossover_rate': tune.uniform(0.025,1),
        'differential_weight': tune.uniform(0.2,1.5),
    }
    
    trainable_with_resources = tune.with_resources(
        run_search, 
        {"cpu": 1, "gpu": gpus_per_trial}
    )
    
    run_config = train.RunConfig(
        storage_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tuning_results'),
        name="de_tuning"
    )
    
    # Early stopping after 50 trials (75% discarded)
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        grace_period=50,
    )
    
    # Tuner Setup
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            metric="goodness",
            mode="min",
            num_samples=1000,
            max_concurrent_trials=max_concurrent_trials,
            scheduler=asha_scheduler
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    # Analysis of Results
    print("\n--- Tuning Complete ---")

    best_result = results.get_best_result()
    print("Best trial config: ", best_result.config)
