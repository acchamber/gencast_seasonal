# code from the gencast_seasonal_test_notebook, put into a python file to submit to the full orchid GPU node

import dataclasses
import datetime
import math
import time
from glob import glob
from typing import Optional
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
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
random_latent_size = 256
random_attention_type = "triblockdiag_mha"
random_mesh_size = 5 #5
random_num_heads = 4 #4
random_attention_k_hop = 8 #8
random_resolution = "1p0"

params = None  # Filled in below
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

example_batch = xarray.load_dataset("/gws/pw/j07/climateresilience/MLdata/glosea_600_slices/source-glosea600_date-19930209_levels-2_lead-3m.nc",engine="netcdf4").compute()

dic = dataclasses.asdict(task_config)
del dic["input_duration"]
print(task_config)
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, input_target_num=[0,1,2],input_target_lead=[1,1,1], 
    **dic)


print("All Examples:  ", example_batch.dims.mapping)

train_inputs = train_inputs.isel(forecastMonth=0) # forecast month just repeated along len direction
train_inputs = train_inputs.expand_dims("time").rename({"longitude":"lon","latitude":"lat","time":"batch"})
train_targets = train_targets.expand_dims("time").rename({"longitude":"lon","latitude":"lat","time":"batch"})
train_inputs = train_inputs.assign_coords(lat=train_inputs.lat.data[::-1])
train_targets = train_targets.assign_coords(lat=train_targets.lat.data[::-1])
train_inputs = train_inputs.assign_coords(lon=train_inputs.lon -.5)
train_targets = train_targets.assign_coords(lon=train_targets.lon -.5)
train_targets = train_targets.expand_dims("number")
train_forcings = train_inputs[['year_progress_sin', 'year_progress_cos']]
train_forcings = train_forcings.isel(number=1).expand_dims("number").assign_coords(number=[2])
train_inputs = train_inputs.fillna(0)
train_targets = train_targets.fillna(0)

train_inputs = train_inputs.transpose("batch",...)
train_targets = train_targets.transpose("batch",...)
train_forcings = train_forcings.transpose("batch",...)

# train_inputs = xarray.concat([train_inputs,train_inputs,train_inputs],dim="batch")
# train_targets =  xarray.concat([train_targets,train_targets,train_targets],dim="batch")
# train_forcings = xarray.concat([train_forcings,train_forcings,train_forcings],dim="batch")



dir_prefix = "/home/users/achamber/gencast_seasonal/"
with open(dir_prefix+"diffs_stddev_by_level.nc","rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(dir_prefix+"mean_by_level.nc","rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with open(dir_prefix+"stddev_by_level.nc","rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()
with open(dir_prefix+"min_by_level.nc","rb") as f:
  min_by_level = xarray.load_dataset(f).compute()

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
def run_forward(inputs, targets_template, forcings):
  predictor = construct_wrapped_gencast()
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


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
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), i, t, f
    )
    return loss, (diagnostics, next_state)

  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True
  )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads


if params is None:
  init_jitted = jax.jit(loss_fn.init)
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets=train_targets,
      forcings=train_forcings,
  )


loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)
grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
# We also produce a pmapped version for running in parallel.
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

grad_fn_pmap = xarray_jax.pmap(grads_fn_jitted, dim="device")

loss, diagnostics = loss_fn_jitted(
    jax.random.PRNGKey(0),
    train_inputs,
    train_targets,
    train_forcings)
print("Loss:", float(loss),flush=True)
print(diagnostics,flush=True)

start_time = time.time()
loss, diagnostics, next_state, grads = grads_fn_jitted(
    params=params,
    state=state,
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings)
mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}",flush=True)
print(f" time:{time.time() - start_time}",flush=True)

def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result
print(memory_usage(),flush=True)
train_inputs2 = train_inputs.expand_dims({"device":2},axis=0)
train_targets2 = train_targets.expand_dims({"device":2},axis=0)
train_forcings2 = train_forcings.expand_dims({"device":2},axis=0)

print("Train Inputs:  ", train_inputs2.dims.mapping)
print("Train Targets: ", train_targets2.dims.mapping)
print("Train Forcings:", train_forcings2.dims.mapping)

start_time = time.time()
loss, diagnostics, next_state, grads = grad_fn_pmap(
    params,
    state,
    train_inputs2,
    train_targets2,
    train_forcings2)
grads = jax.lax.pmean(grads, axis_name='device')
mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}",flush=True)
print(f" time:{time.time() - start_time}",flush=True)


