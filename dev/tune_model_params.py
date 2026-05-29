"""
Runs hyperparameter search for the model parameters.
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

seed = 17
wrkdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

print(f"Is CUDA available: {torch.cuda.is_available()}")

def run_search(config, cortex, E, ROI, stimulator):
    from ABL import LocalizationSession
    import torch
    import random
    import os
    from ray import tune

    def calculate_dist(S):
        """Helper function to calculate the distance between estimated and true source."""
        log_prob = S.response_model.log_source_prob
        estimate_ind = torch.argmax(log_prob)
        true_ind = S.source['index']
        distance = ((S.mesh.vertices[S.mesh.ROI[true_ind], :] - S.mesh.vertices[S.mesh.ROI[estimate_ind], :])**2).sum().sqrt()
        return distance * 1000

    def calculate_FS_RMSE(S):
        """Helper function to calculate source parameter accuracy between estimated and true source"""

        def directional_sigmoid(e_mag, angles, params):
            # The translation term is direction-dependent, modeled by a Fourier series
            directional_translation = (
                params['rf_translate']['values'] +
                params['rf_fourier1']['values'] * torch.cos(angles) +
                params['rf_fourier2']['values'] * torch.sin(angles) +
                params['rf_fourier3']['values'] * torch.cos(angles * 2) +
                params['rf_fourier4']['values'] * torch.sin(angles * 2)
            )
            inner_term = torch.clamp(-params['rf_slope']['values'] * (e_mag - directional_translation), -16, 16)
            response_prob = e_mag / (e_mag + torch.exp(inner_term))
            return torch.clamp(response_prob, S.response_model.prob_lim[0], S.response_model.prob_lim[1])
        true_param_dict = S.source['rf_params']
        true_source_ind = S.source['index']

        estimated_param_dict_full = S.response_model.param_dict
        estimated_param_dict = {key: {'values': value['values'][true_source_ind].cpu()} for key,value in estimated_param_dict_full.items()}

        # Defined range where the function is evaluated at
        num_e_steps = 30
        num_angle_steps = 36
        e_mag_range = torch.linspace(0.1,200,num_e_steps)
        angle_range = torch.linspace(0, 2*torch.pi,num_angle_steps)

        e_mag_grid, angles_grid = torch.meshgrid(e_mag_range, angle_range, indexing='ij')

        # Compute response surfaces
        with torch.no_grad(): # Ensure no gradients are computed
            true_response = directional_sigmoid(e_mag_grid, angles_grid, true_param_dict)
            estimated_response = directional_sigmoid(e_mag_grid, angles_grid, estimated_param_dict)

        fs_rmse = torch.sqrt(torch.mean((true_response - estimated_response)**2))

        return fs_rmse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    save_path = os.path.join(wrkdir, "examples", "saved", "example_subject")

    # Define all source scenarios
    source_ind_scenarios = [2297, 5907, 4504, 2333, 1921,3648,4884,6801,7633,2691]
    true_rf_param_dict1 = {
        "rf_slope": dict(values=torch.tensor([0.11])), "rf_translate": dict(values=torch.tensor([109.4])),
        "rf_fourier1": dict(values=torch.tensor([-9.9])), "rf_fourier2": dict(values=torch.tensor([15.2])),
        "rf_fourier3": dict(values=torch.tensor([-22.8])), "rf_fourier4": dict(values=torch.tensor([22.4])),
    }
    true_rf_param_dict2 = {
        "rf_slope": dict(values=torch.tensor([0.09])), "rf_translate": dict(values=torch.tensor([76.8])),
        "rf_fourier1": dict(values=torch.tensor([13.4])), "rf_fourier2": dict(values=torch.tensor([-18.1])),
        "rf_fourier3": dict(values=torch.tensor([21.0])), "rf_fourier4": dict(values=torch.tensor([2.1])),
    }
    true_rf_param_dict3 = {
        "rf_slope": dict(values=torch.tensor([0.08])), "rf_translate": dict(values=torch.tensor([135.9])),
        "rf_fourier1": dict(values=torch.tensor([17.4])), "rf_fourier2": dict(values=torch.tensor([-21.5])),
        "rf_fourier3": dict(values=torch.tensor([11.8])), "rf_fourier4": dict(values=torch.tensor([-12.1])),
    }
    true_rf_param_dict4 = {
        "rf_slope": dict(values=torch.tensor([0.12])), "rf_translate": dict(values=torch.tensor([99.2])),
        "rf_fourier1": dict(values=torch.tensor([-22.0])), "rf_fourier2": dict(values=torch.tensor([8.0])),
        "rf_fourier3": dict(values=torch.tensor([14.0])), "rf_fourier4": dict(values=torch.tensor([-17.9])),
    }
    true_rf_param_dict5 = {
        "rf_slope": dict(values=torch.tensor([0.06])), "rf_translate": dict(values=torch.tensor([116.3])),
        "rf_fourier1": dict(values=torch.tensor([-16.3])), "rf_fourier2": dict(values=torch.tensor([-17.6])),
        "rf_fourier3": dict(values=torch.tensor([20.4])), "rf_fourier4": dict(values=torch.tensor([6.8])),
    }
    true_rf_param_dict6 = {
        "rf_slope":dict(values=torch.tensor([0.09])),"rf_translate":dict(values=torch.tensor([77.2])),
        "rf_fourier1":dict(values=torch.tensor([-7.0])),"rf_fourier2":dict(values=torch.tensor([4.4])),  
        "rf_fourier3":dict(values=torch.tensor([0.7])),"rf_fourier4":dict(values=torch.tensor([8.5])),          
    }
    true_rf_param_dict7 = {
        "rf_slope":dict(values=torch.tensor([0.10])),"rf_translate":dict(values=torch.tensor([114.5])),
        "rf_fourier1":dict(values=torch.tensor([-9.7])),"rf_fourier2":dict(values=torch.tensor([-8.7])),  
        "rf_fourier3":dict(values=torch.tensor([6.1])),"rf_fourier4":dict(values=torch.tensor([5.1])),          
    }
    true_rf_param_dict8 = {
        "rf_slope":dict(values=torch.tensor([0.11])),"rf_translate":dict(values=torch.tensor([149.8])),
        "rf_fourier1":dict(values=torch.tensor([-23.3])),"rf_fourier2":dict(values=torch.tensor([-18.9])),  
        "rf_fourier3":dict(values=torch.tensor([7.4])),"rf_fourier4":dict(values=torch.tensor([-5.9])),          
    }
    true_rf_param_dict9 = {
        "rf_slope":dict(values=torch.tensor([0.08])),"rf_translate":dict(values=torch.tensor([59.9])),
        "rf_fourier1":dict(values=torch.tensor([-13.5])),"rf_fourier2":dict(values=torch.tensor([-1.1])),  
        "rf_fourier3":dict(values=torch.tensor([-15.2])),"rf_fourier4":dict(values=torch.tensor([0.0])),          
    }
    true_rf_param_dict10 = {
        "rf_slope":dict(values=torch.tensor([0.07])),"rf_translate":dict(values=torch.tensor([87.9])),
        "rf_fourier1":dict(values=torch.tensor([-1.9])),"rf_fourier2":dict(values=torch.tensor([-18.2])),  
        "rf_fourier3":dict(values=torch.tensor([-18.4])),"rf_fourier4":dict(values=torch.tensor([-19.7])),          
    }
    true_rf_param_dicts = [true_rf_param_dict1,true_rf_param_dict2,true_rf_param_dict3,true_rf_param_dict4,true_rf_param_dict5,true_rf_param_dict6,true_rf_param_dict7,true_rf_param_dict8,true_rf_param_dict9,true_rf_param_dict10]

    num_scenarios = len(source_ind_scenarios)

    # Run each scenario
    total_penalty = 0

    for j in range(num_scenarios):
        source_ind = source_ind_scenarios[j]
        source_param_dict = true_rf_param_dicts[j]

        # Instantiate session
        S = LocalizationSession(cortex=cortex, save_path=save_path, stimulator=stimulator, efield_set=E, ROI=ROI, device=device, silent_mode=True)

        # Generate source and set its true parameters for the current scenario
        S.generate_source([source_ind])
        S.source['rf_params'] = source_param_dict

        # Apply the hyperparameters for this trial from the config
        S.response_model.opt_settings = {
            'general': {'lr': 0.001, 'gamma': config['gamma'], 'global_gamma': config['global_gamma'], 'clip_grad': 0.1, 'betas': [config['beta0'], config['beta1']], 'opt_steps': 150},
            'rf_slope': {'lr': config['lr_slope']}, 'rf_translate': {'lr': config['lr_translate']},
            'rf_fourier1': {'lr': config['lr_fourier']}, 'rf_fourier2': {'lr': config['lr_fourier']},
            'rf_fourier3': {'lr': config['lr_fourier']}, 'rf_fourier4': {'lr': config['lr_fourier']}
        }
        S.response_model.regularization_lambda = config['reg_lmbd']

        trial_count = 50
        dists = torch.ones((trial_count,))

        # Run the simulation loop for the current scenario
        for i in range(S.trial_number, trial_count):
            S.start_trial()
            didt = S.choose_next_stimulus()
            S.record_efield_trial(didt)
            S.simulate_response()
            S.localize()
            S.complete_trial()
            dists[i] = calculate_dist(S)

        # Calculate penalty for the scenario
        scenario_penalty = calculate_FS_RMSE(S)

        # Report the penalty so far
        total_penalty += scenario_penalty.cpu().item()
        tune.report({"accuracy": total_penalty})

        # Clean up
        S.end_session()
        del S

if __name__ == "__main__":
    model = os.path.join(wrkdir, "examples", "saved", "example_subject", "example_model.pkl")
    stimulator_path = os.path.join(wrkdir, "examples", "coils", "5coil_example.json")

    paths_to_check = {
        "E-Field CSV": model,
        "Stimulator JSON": stimulator_path
    }

    for name, path in paths_to_check.items():
        print(f"Checking for {name}: {path} ... {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
    print("----------------------------")

    E, ROI, cortex = preprocessing.load_example_model(model)
    stimulator = preprocessing.get_stimulator_from_file(stimulator_path)

    n_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count()
    else:
        n_cpus = int(n_cpus)

    n_gpus = os.getenv('SLURM_GPUS')
    if n_gpus is None:
        n_gpus = 1
    else:
        n_gpus = int(n_gpus)

    print(f"SLURM allocated {n_cpus} CPUs and {n_gpus} GPUs.")

    ray.init(num_cpus=n_cpus, num_gpus=n_gpus)
    # Put large data into Ray's object store to avoid repeated serialization
    E_ref = ray.put(E)
    ROI_ref = ray.put(ROI)
    cortex_ref = ray.put(cortex)
    stimulator_ref = ray.put(stimulator)

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

    # Hyperparameter Search Space. Note: the ranges here are just examples and 
    # should be adjusted based on domain knowledge and preliminary testing.
    search_space = {
        "lr_slope": tune.uniform(0.001,0.005),
        "lr_translate": tune.uniform(0.0001, 0.002),
        "lr_fourier": tune.uniform(0.001,0.005),
        "gamma": tune.uniform(0.9, 1.0),
        "global_gamma": tune.uniform(0.9, 1.0),
        "reg_lmbd": tune.uniform(0.01, 0.1),
        "beta0": tune.uniform(0.3, 1.0),
        "beta1": tune.uniform(0.3, 1.0),
    }

    trainable_with_resources = tune.with_resources(
        run_search,
        {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    )

    trainable_with_data = tune.with_parameters(
        trainable_with_resources,
        cortex=cortex_ref,
        E=E_ref,
        ROI=ROI_ref,
        stimulator=stimulator_ref
    )
    
    run_config = tune.RunConfig(
        storage_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tuning_results'),
        name="generalized_tuning_random_sample"
    )

    # Early stopping after 5 scenarios (75% discarded)
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        grace_period=5,
    )

    # Tuner Setup
    tuner = tune.Tuner(
        trainable_with_data,
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(),
            metric="accuracy",
            mode="min",
            num_samples=4,
            max_concurrent_trials=max_concurrent_trials,
            scheduler=asha_scheduler
        ),
        param_space=search_space,
        run_config=run_config
    )
    results = tuner.fit()

    # Analysis of Results
    print("\n--- Tuning Complete ---")
    
    best_result = results.get_best_result()
    print("Best trial config: ", best_result.config)
