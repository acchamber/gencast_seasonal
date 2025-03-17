# Training loop for gencast_seasonal, designed for a 4 A100 node on JASMIN

import dataclasses
import datetime
import math
import random
import time
import reprlib
from glob import glob
from typing import Optional
import haiku as hk
import jax
import optax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import jax.numpy as jnp
import xarray

from gencast_seasonal import rollout
from gencast_seasonal import xarray_jax
from gencast_seasonal import normalization
from gencast_seasonal import checkpoint
from gencast_seasonal import data_utils
from gencast_seasonal import xarray_tree
from gencast_seasonal import gencast
from gencast_seasonal import denoiser
from gencast_seasonal import nan_cleaning 


input_path = "/gws/pw/j07/climateresilience/MLdata"

latent_value_options = [int(2**i) for i in range(4, 10)]
random_latent_size = 256 # 512 base, testing speedup from lower
random_attention_type = "triblockdiag_mha"
random_mesh_size = 5 #5 base, testing speedup from lower
random_num_heads = 4 #4
random_attention_k_hop = 8 #16
random_resolution = "1p0"

load_checkpoint = False # False if new run, True if contining
state = {}
task_config = gencast.TASK
# Use default values.
sampler_config = gencast.SamplerConfig()
noise_config = gencast.NoiseConfig()
noise_encoder_config = denoiser.NoiseEncoderConfig()
# Configure, otherwise use default values.
denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
sparse_transformer_config = denoiser.SparseTransformerConfig(
    attention_k_hop=random_attention_k_hop,
    attention_type=random_attention_type,
    d_model=random_latent_size,
    num_heads=random_num_heads
    ),
mesh_size=random_mesh_size,
latent_size=random_latent_size,
)

source_files = glob(f"{input_path}/glosea_600_slices/source*.nc")

def get_training_data(file_path,lead_time=1,n_models_total=3,batch_n=1,fixed=False):
    
    raw_training = xarray.load_dataset(file_path,engine="netcdf4").compute()
    raw_training.assign_coords(batch=[batch_n]) # remove datetime from batch
    dic = dataclasses.asdict(task_config)
    del dic["input_duration"] # easier than removing input_duration from the task config
    models_picked=n_models_total
    lead_time_list = [lead_time]*models_picked
    input_list = random.sample(range(0,raw_training.sizes["number"]),models_picked)
    if fixed == True:
        input_list = [0,1,2]
    
    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    raw_training, input_target_num=input_list,input_target_lead=lead_time_list, 
    **dic)
    
    train_inputs = train_inputs.isel(forecastMonth=0) # forecast month just repeated along len direction
    train_inputs = train_inputs.expand_dims("time").rename({"longitude":"lon","latitude":"lat","time":"batch"})
    train_targets = train_targets.expand_dims("time").rename({"longitude":"lon","latitude":"lat","time":"batch"})
    train_inputs = train_inputs.assign_coords(lat=train_inputs.lat.data[::-1])
    train_targets = train_targets.assign_coords(lat=train_targets.lat.data[::-1])
    train_inputs = train_inputs.assign_coords(lon=train_inputs.lon -.5)
    train_targets = train_targets.assign_coords(lon=train_targets.lon -.5)
    train_targets = train_targets.expand_dims("number")
    train_forcings = train_inputs[['year_progress_sin', 'year_progress_cos']]
    train_forcings = train_forcings.sel(number=input_list[0]).expand_dims("number").assign_coords(number=[input_list[-1]])
    train_inputs = train_inputs.fillna(0)
    train_targets = train_targets.fillna(0)
    train_forcings = train_forcings.fillna(0)

    return train_inputs, train_targets, train_forcings


def construct_wrapped_gencast():
  """Constructs and wraps the GenCast Predictor."""
  predictor = gencast.GenCast(
      sampler_config=sampler_config,
      task_config=task_config,
      denoiser_architecture_config=denoiser_architecture_config,
      noise_config=noise_config,
      noise_encoder_config=noise_encoder_config,
  )

  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level,
  )

  predictor = nan_cleaning.NaNCleaner(
      predictor=predictor,
      reintroduce_nans=True,
      fill_value=min_by_level,
      var_to_clean='sea_surface_temperature',
  )

  return predictor

@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
  predictor = construct_wrapped_gencast()
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics),
  )

def grads_fn(params, state, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, jax.random.PRNGKey(0), i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

def setup_optimizer():
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(
    init_value = 0.01,
    peak_value = .03,
    warmup_steps = 1000,
    decay_steps = 1000000,
    end_value = 0.0,
    exponent = 0.1,)
    #lr = 0.001
    
    optimiser = optax.adam(lr_schedule, b1=0.9, b2=0.999, eps=1e-8)
    opt_state = optimiser.init(params)
    return optimiser, opt_state

loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)

grads_fn_jitted = jax.jit(grads_fn)

train_inputs, train_targets, train_forcings = get_training_data(source_files[0])

dir_prefix = "/home/users/achamber/gencast_seasonal/"
with open(dir_prefix+"diffs_stddev_by_level.nc","rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).astype(np.float32).compute()
    
mean_by_level = xarray.load_dataset(input_path+"/glosea_600_hc_mean_by_level.nc").compute()

stddev_by_level = xarray.load_dataset(input_path+"/glosea_600_hc_stddev_by_level.nc").compute()

with open(dir_prefix+"min_by_level.nc","rb") as f:
  min_by_level = xarray.load_dataset(f).astype(np.float32).compute()

if load_checkpoint is False:
      params = None
      init_jitted = jax.jit(loss_fn.init)
      params, state = init_jitted(
          rng=jax.random.PRNGKey(0),
          inputs=train_inputs,
          targets=train_targets,
          forcings=train_forcings,
      )


optimiser, opt_state = setup_optimizer()
total_time = time.time()
starting_x = 1
for x in range(starting_x,starting_x+260):
    print(f"training attempt {x}",flush=True)
    start_time = time.time()
    
    train_inputs, train_targets, train_forcings = get_training_data(source_files[x])
    # train_inputs2, train_targets2, train_forcings2 =  get_training_data(source_files[x])
    # train_inputs = xarray.concat([train_inputs,train_inputs2],dim="batch")
    # train_targets =  xarray.concat([train_targets,train_targets2],dim="batch")
    # train_forcings = xarray.concat([train_forcings,train_forcings2],dim="batch")
    
    print(f"{time.time() - start_time} seconds to prepare batch",flush=True)
    loss, diagnostics, next_state, grads = grads_fn_jitted(params, state, train_inputs, train_targets, train_forcings)
    mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
    print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}",flush=True)
    print(diagnostics,flush=True)
    print(f"{time.time() - start_time} seconds to calculate gradients",flush=True)
    updates, opt_state = optimiser.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(f"{time.time() - start_time} seconds to update gradients",flush=True)

print(f"{time.time() - total_time} seconds total for 260 updates",flush=True)

def flatten_dict(d, parent_key='', sep='//'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_model_params(d, file_path):
    flat_dict = flatten_dict(d)
    # Convert JAX arrays to NumPy for saving
    np_dict = {k: np.array(v) if isinstance(v, jnp.ndarray) else v for k, v in flat_dict.items()}
    np.savez(file_path, **np_dict)

save_model_params(params, "training_test_params.npz")