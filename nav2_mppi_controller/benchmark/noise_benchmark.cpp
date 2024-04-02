// Copyright (c) 2023 AOS
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
//
#include <benchmark/benchmark.h>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xtensor.hpp>
#include "gtest/gtest.h"

static void BM_Noise_xt_random(benchmark::State & state)
{
    xt::xtensor<float, 2> noises_vx_;
    for (auto _ : state) {
        noises_vx_ = xt::eval(xt::random::randn<float>({2000, 56}, 0, 0.2));
    }
}

static void BM_Noise_xt_random_noalias(benchmark::State & state)
{
    xt::xtensor<float, 2> noises_vx_;
    for (auto _ : state) {
        xt::noalias(noises_vx_) = xt::eval(xt::random::randn<float>({2000, 56}, 0, 0.2));
    }
}

static void BM_Noise_adapt_vector_1k(benchmark::State & state)
{
    std::vector<float> v(1000 * 2000 * 56);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.2);
    for (auto& el : v) {
        el = d(gen);
    }
    auto a = xt::adapt(v, {1000, 2000, 56});
    xt::xtensor<float, 2> noises_vx_;
    int i = 0;
    for (auto _ : state) {
        // Create a view for the current row with shape {2000, 56}
        i = (i + 1) % 1000;
        noises_vx_ = xt::eval(xt::view(a, i, xt::all(), xt::all()));
    }
}

static void BM_Noise_adapt_vector_3k(benchmark::State & state)
{
    std::vector<float> v(3000 * 2000 * 56);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.2);
    for (auto& el : v) {
        el = d(gen);
    }
    auto a = xt::adapt(v, {3000, 2000, 56});
    xt::xtensor<float, 2> noises_vx_;
    int i = 0;
    for (auto _ : state) {
        // Create a view for the current row with shape {2000, 56}
        i = (i + 1) % 3000;
        noises_vx_ = xt::eval(xt::view(a, i, xt::all(), xt::all()));
    }
}

static void BM_Noise_adapt_vector_shape_1k(benchmark::State & state)
{
    // These are int not size_t as they come from parameters IRL.
    int noise_pregenerate_size = 1000;
    int batch_size = 2000;
    int time_steps = 56;

    std::vector<float> v(noise_pregenerate_size * batch_size * time_steps);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.2);
    for (auto& el : v) {
        el = d(gen);
    }

    xt::xtensor<float, 2> noises_vx_;

    int i = 0;
    for (auto _ : state) {
        // Create a view for the current row with shape {2000, 56}
        xt::static_shape<std::size_t, 3> sh = {
          static_cast<size_t>(noise_pregenerate_size),
          static_cast<size_t>(batch_size),
          static_cast<size_t>(time_steps)};
        auto a = xt::adapt(v, sh);
        i = (i + 1) % 1000;
        noises_vx_ = xt::eval(xt::view(a, i, xt::all(), xt::all()));
    }
}

static void BM_Noise_adapt_vector_shape_3k(benchmark::State & state)
{
    // These are int not size_t as they come from parameters IRL.
    int noise_pregenerate_size = 3000;
    int batch_size = 2000;
    int time_steps = 56;

    std::vector<float> v(noise_pregenerate_size * batch_size * time_steps);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.2);
    for (auto& el : v) {
        el = d(gen);
    }

    xt::xtensor<float, 2> noises_vx_;

    int i = 0;
    for (auto _ : state) {
        // Create a view for the current row with shape {2000, 56}
        xt::static_shape<std::size_t, 3> sh = {
          static_cast<size_t>(noise_pregenerate_size),
          static_cast<size_t>(batch_size),
          static_cast<size_t>(time_steps)};
        auto a = xt::adapt(v, sh);
        i = (i + 1) % 1000;
        noises_vx_ = xt::eval(xt::view(a, i, xt::all(), xt::all()));
    }
}

static void BM_Noise_adapt_vector_shape_3k_adopt_once(benchmark::State & state)
{
    // These are int not size_t as they come from parameters IRL.
    int noise_pregenerate_size = 3000;
    int batch_size = 2000;
    int time_steps = 56;

    std::vector<float> v(noise_pregenerate_size * batch_size * time_steps);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0, 0.2);
    for (auto& el : v) {
        el = d(gen);
    }

    xt::static_shape<std::size_t, 3> sh = {
      static_cast<size_t>(noise_pregenerate_size),
      static_cast<size_t>(batch_size),
      static_cast<size_t>(time_steps)};

    // If we could figure out what this type is, while
    // still being dynamic we could adapt only once.
    auto a = xt::adapt(v, sh);
    xt::xtensor<float, 2> noises_vx_;

    int i = 0;
    for (auto _ : state) {
        // Create a view for the current row with shape {2000, 56}
        i = (i + 1) % 1000;
        noises_vx_ = xt::eval(xt::view(a, i, xt::all(), xt::all()));
    }
}

BENCHMARK(BM_Noise_xt_random);
BENCHMARK(BM_Noise_xt_random_noalias);
BENCHMARK(BM_Noise_adapt_vector_1k);
BENCHMARK(BM_Noise_adapt_vector_3k);
BENCHMARK(BM_Noise_adapt_vector_shape_1k);
BENCHMARK(BM_Noise_adapt_vector_shape_3k);
BENCHMARK(BM_Noise_adapt_vector_shape_3k_adopt_once);

BENCHMARK_MAIN();
