# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Denoising diffusion models based on the framework of [1].

Throughout we will refer to notation and equations from [1].

  [1] Elucidating the Design Space of Diffusion-Based Generative Models
  Karras, Aittala, Aila and Laine, 2022
  https://arxiv.org/abs/2206.00364
"""

from typing import Any, Optional, Tuple

import chex
from gencast_seasonal import casting
from gencast_seasonal import denoiser
from gencast_seasonal import dpm_solver_plus_plus_2s
from gencast_seasonal import graphcast
from gencast_seasonal import losses
from gencast_seasonal import predictor_base
from gencast_seasonal import samplers_utils
from gencast_seasonal import xarray_jax
import haiku as hk
import jax
import xarray


TARGET_SURFACE_VARS = (
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind', 
    'total_precipitation_month',
    #'sea_surface_temperature',
)

TARGET_SURFACE_NO_PRECIP_VARS = (
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    #'sea_surface_temperature',
)

TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    #"specific_humidity",
)

PRESSURE_LEVELS_SEASONAL_2 = (500, 850)

GENERATED_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
)

STATIC_VARS = (
    "geopotential_at_surface",
    "land_sea_mask",
)

TASK = graphcast.TaskConfig(
    input_variables=(
        # GenCast doesn't take precipitation as input.
        TARGET_SURFACE_VARS
        + TARGET_ATMOSPHERIC_VARS
        + GENERATED_FORCING_VARS
        + STATIC_VARS
    ),
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    # GenCast Seasonal only uses year_progress
    forcing_variables=GENERATED_FORCING_VARS,
    # Only using 2 pressure levels at first
    pressure_levels=PRESSURE_LEVELS_SEASONAL_2,
    
    input_duration='24h', # Change this
)


@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
  """Configures the sampler used to draw samples from GenCast.

      max_noise_level: The highest noise level used at the start of the
        sequence of reverse diffusion steps.
      min_noise_level: The lowest noise level used at the end of the sequence of
        reverse diffusion steps.
      num_noise_levels: Determines the number of noise levels used and hence the
        number of reverse diffusion steps performed.
      rho: Parameter affecting the spacing of noise steps. Higher values will
        concentrate noise steps more around zero.
      stochastic_churn_rate: S_churn from the paper. This controls the rate
        at which noise is re-injected/'churned' during the sampling algorithm.
        If this is set to zero then we are performing deterministic sampling
        as described in Algorithm 1.
      churn_max_noise_level: Maximum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      churn_min_noise_level: Minimum noise level at which stochastic churn
        occurs. S_min from the paper. Only used if stochastic_churn_rate > 0.
      noise_level_inflation_factor: This can be used to set the actual amount of
        noise injected higher than what the denoiser is told has been added.
        The motivation is to compensate for a tendency of L2-trained denoisers
        to remove slightly too much noise / blur too much. S_noise from the
        paper. Only used if stochastic_churn_rate > 0.
  """
  max_noise_level: float = 88.
  min_noise_level: float = 0.02
  num_noise_levels: int = 20
  rho: float = 7.
  # Stochastic sampler settings.
  stochastic_churn_rate: float = 2.5
  churn_min_noise_level: float = 0.75
  churn_max_noise_level: float = float('inf')
  noise_level_inflation_factor: float = 1.05


@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
  training_noise_level_rho: float = 7.0
  training_max_noise_level: float = 88.0
  training_min_noise_level: float = 0.02


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
  description: str
  license: str
  params: dict[str, Any]
  task_config: graphcast.TaskConfig
  denoiser_architecture_config: denoiser.DenoiserArchitectureConfig
  sampler_config: SamplerConfig
  noise_config: NoiseConfig
  noise_encoder_config: denoiser.NoiseEncoderConfig


class GenCast(predictor_base.Predictor):
  """Predictor for a denoising diffusion model following the framework of [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364

  Unlike the paper, we have a conditional model and our denoising function
  conditions on previous timesteps.

  As the paper demonstrates, the sampling algorithm can be varied independently
  of the denoising model and its training procedure, and it is separately
  configurable here.
  """

  def __init__(
      self,
      task_config: graphcast.TaskConfig,
      denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
      sampler_config: Optional[SamplerConfig] = None,
      noise_config: Optional[NoiseConfig] = None,
      noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None,
  ):
    """Constructs GenCast."""
    # Output size depends on number of variables being predicted.
    num_surface_vars = len(
        set(task_config.target_variables)
        - set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_atmospheric_vars = len(
        set(task_config.target_variables)
        & set(graphcast.ALL_ATMOSPHERIC_VARS)
    )
    num_outputs = (
        num_surface_vars
        + len(task_config.pressure_levels) * num_atmospheric_vars
    )
    denoiser_architecture_config.node_output_size = num_outputs
    self._denoiser = denoiser.Denoiser(
        noise_encoder_config,
        denoiser_architecture_config,
    )
    self._sampler_config = sampler_config
    # Singleton to avoid re-initializing the sampler for each inference call.
    self._sampler = None
    self._noise_config = noise_config

  def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the noisy targets input to the underlying network."""
    return (noise_scale**2 + 1)**-0.5

  def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the underlying network's raw outputs."""
    return noise_scale * (noise_scale**2 + 1)**-0.5

  def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    """Scaling applied to the skip connection."""
    return 1 / (noise_scale**2 + 1)

  def _loss_weighting(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
    r"""The loss weighting \lambda(\sigma) from the paper."""
    return self._c_out(noise_scale) ** -2

  def _preconditioned_denoiser(
      self,
      inputs: xarray.Dataset,
      noisy_targets: xarray.Dataset,
      noise_levels: xarray.DataArray,
      forcings: Optional[xarray.Dataset] = None,
      **kwargs) -> xarray.Dataset:
    """The preconditioned denoising function D from the paper (Eqn 7)."""
    raw_predictions = self._denoiser(
        inputs=inputs,
        noisy_targets=noisy_targets * self._c_in(noise_levels),
        noise_levels=noise_levels,
        forcings=forcings,
        **kwargs)
    return (raw_predictions * self._c_out(noise_levels) +
            noisy_targets * self._c_skip(noise_levels))

  def loss_and_predictions(
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: Optional[xarray.Dataset] = None,
  ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
    return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: Optional[xarray.Dataset] = None,
           ) -> predictor_base.LossAndDiagnostics:

    if self._noise_config is None:
      raise ValueError('Noise config must be specified to train GenCast.')

    # Sample noise levels:
    dtype = casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    key = hk.next_rng_key()
    batch_size = inputs.sizes['batch']
    noise_levels = xarray_jax.DataArray(
        data=samplers_utils.rho_inverse_cdf(
            min_value=self._noise_config.training_min_noise_level,
            max_value=self._noise_config.training_max_noise_level,
            rho=self._noise_config.training_noise_level_rho,
            cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype)),
        dims=('batch',))

    # Sample noise and apply it to targets:
    noise = (
        samplers_utils.spherical_white_noise_like(targets) * noise_levels
    )
    noisy_targets = targets + noise
    denoised_predictions = self._preconditioned_denoiser(
        inputs, noisy_targets, noise_levels, forcings)
    loss, diagnostics = losses.weighted_mse_per_level(
        denoised_predictions,
        targets,
        # Weights are same as we used for GraphCast.
        per_variable_weights={
            # Any variables not specified here are weighted as 1.0.
            # A single-level variable, but an important headline variable
            # and also one which we have struggled to get good performance
            # on at short lead times, so leaving it weighted at 1.0, equal
            # to the multi-level variables:
            '2m_temperature': 1.0,
            # MSLP key for NAO prediction and monthly forecasting via weather regimes, so overweigh it 
            'mean_sea_level_pressure': 1.5,
        },
    )
    loss *= self._loss_weighting(noise_levels)
    return loss, diagnostics

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: Optional[xarray.Dataset] = None,
               **kwargs) -> xarray.Dataset:
    if self._sampler_config is None:
      raise ValueError(
          'Sampler config must be specified to run inference on GenCast.'
      )
    if self._sampler is None:
      self._sampler = dpm_solver_plus_plus_2s.Sampler(
          self._preconditioned_denoiser, **self._sampler_config
      )
    return self._sampler(inputs, targets_template, forcings, **kwargs)
