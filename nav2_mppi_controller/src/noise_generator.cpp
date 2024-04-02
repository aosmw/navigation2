// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/tools/noise_generator.hpp"

#include <memory>
#include <mutex>
#include <fstream>
#include <iostream>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xcsv.hpp>


namespace mppi
{

void NoiseGenerator::initialize(
  mppi::models::OptimizerSettings & settings, bool is_holonomic,
  const std::string & name, ParametersHandler * param_handler)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;
  active_ = true;

  auto getParam = param_handler->getParamGetter(name);
  getParam(regenerate_noises_, "regenerate_noises", false);
  getParam(dump_noises_, "dump_noises", true);
  getParam(noise_seed_, "noise_seed", 0);
  getParam(noise_pregenerate_size_, "noise_pregenerate_size", 1000);

  if (noise_seed_ != 0) {
    xt::random::seed(noise_seed_);
  }

  if (noise_pregenerate_size_ != 0) {
    preGenerateNoisedControls();
  } else {
    if (regenerate_noises_) {
      noise_thread_ = std::thread(std::bind(&NoiseGenerator::noiseThread, this));
    } else {
      generateNoisedControls();
    }
  }
}

void NoiseGenerator::shutdown()
{
  active_ = false;
  ready_ = true;
  noise_cond_.notify_all();
  if (noise_thread_.joinable()) {
    noise_thread_.join();
  }
}

void NoiseGenerator::generateNextNoises()
{
  // Trigger the thread to run in parallel to this iteration
  // to generate the next iteration's noises (if applicable).
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    ready_ = true;
  }
  noise_cond_.notify_all();
}

void NoiseGenerator::setNoisedControls(
  models::State & state,
  const models::ControlSequence & control_sequence)
{
  std::unique_lock<std::mutex> guard(noise_lock_);
  auto & s = settings_;

  if (noise_pregenerate_size_ != 0)
  {
    xt::static_shape<std::size_t, 3> sh =
      {(size_t)noise_pregenerate_size_, (size_t)s.batch_size, (size_t)s.time_steps};
    auto a = xt::adapt(pregenerated_noise_, sh);
    noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
    xt::noalias(noises_vx_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
    xt::noalias(noises_wz_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    if (is_holonomic_) {
      noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
      xt::noalias(noises_vy_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    }
  }
  xt::noalias(state.cvx) = control_sequence.vx + noises_vx_;
  xt::noalias(state.cwz) = control_sequence.wz + noises_wz_;
  if (is_holonomic_) {
    xt::noalias(state.cvy) = control_sequence.vy + noises_vy_;
  }
}

void NoiseGenerator::reset(mppi::models::OptimizerSettings & settings, bool is_holonomic)
{
  settings_ = settings;
  is_holonomic_ = is_holonomic;

  // Recompute the noises on reset, initialization, and fallback
  {
    std::unique_lock<std::mutex> guard(noise_lock_);
    xt::noalias(noises_vx_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    xt::noalias(noises_vy_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    xt::noalias(noises_wz_) = xt::zeros<float>({settings_.batch_size, settings_.time_steps});
    ready_ = true;
  }

  if (noise_pregenerate_size_!= 0) {
    preGenerateNoisedControls();
  } else {
    if (regenerate_noises_) {
      noise_cond_.notify_all();
    } else {
      generateNoisedControls();
    }
  }
}

void NoiseGenerator::noiseThread()
{
  do {
    std::unique_lock<std::mutex> guard(noise_lock_);
    noise_cond_.wait(guard, [this]() {return ready_;});
    ready_ = false;
    generateNoisedControls();
  } while (active_);
}

void NoiseGenerator::preGenerateNoisedControls()
{
  auto & s = settings_;

  pregenerated_noise_.resize(noise_pregenerate_size_ * s.batch_size * s.time_steps);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> d(0.0f, s.sampling_std.vx);
  for (auto& el : pregenerated_noise_) {
      el = d(gen);
  }
  noise_pregenerate_idx_ = 0;
}

void NoiseGenerator::generateNoisedControls()
{
  auto & s = settings_;

  if (noise_pregenerate_size_ != 0)
  {
    xt::static_shape<std::size_t, 3> sh =
      {(size_t)noise_pregenerate_size_, (size_t)s.batch_size, (size_t)s.time_steps};
    auto a = xt::adapt(pregenerated_noise_, sh);
    noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
    xt::noalias(noises_vx_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
    xt::noalias(noises_wz_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    if (is_holonomic_) {
      noise_pregenerate_idx_ = (noise_pregenerate_idx_ + 1) % noise_pregenerate_size_;
      xt::noalias(noises_wz_) = xt::view(a, noise_pregenerate_idx_, xt::all(), xt::all());
    }
  } else {
    xt::noalias(noises_vx_) = xt::random::randn<float>(
      {s.batch_size, s.time_steps}, 0.0f,
      s.sampling_std.vx);
    xt::noalias(noises_wz_) = xt::random::randn<float>(
      {s.batch_size, s.time_steps}, 0.0f,
      s.sampling_std.wz);
    if (is_holonomic_) {
      xt::noalias(noises_vy_) = xt::random::randn<float>(
        {s.batch_size, s.time_steps}, 0.0f,
        s.sampling_std.vy);
    }
  }

  if (dump_noises_) {
    std::string fn("/tmp/mppi_noises_");
    std::ofstream f(fn + "vx_" + std::to_string(s.sampling_std.vx) + ".csv",
      std::ios::out | std::ios::trunc);
    xt::dump_csv(f, noises_vx_);
    dump_noises_ = false;
  }
}

}  // namespace mppi
