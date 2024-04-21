/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.data

import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import androidx.annotation.Nullable

data class standup_judge(
    var Previous_status_standup:Boolean = false,
    var Count_Stand_up:Int = 0,
    var time_interval_list: MutableList<Double> = mutableListOf(),

    var index: Int = 2 * 2 ,//2的倍數(因為要左右腳)
    var current_index: Int = 0,
    var stand_angle_threshold: Double = 145.0,
    var sit_angle_threshold: Double = 110.0,
    var time_interval_threshold: Double = 0.6, //若高於threshold才算起立
    var MLscore_threshold: Double = 0.4,

    //開始倒數要用的
    var secondsElapsed:Int = 0,
    val handler:Handler = Handler(Looper.getMainLooper()),
    var runnable: Runnable = Runnable {},
    var standup_judge_flag:Boolean = false,

    //過程倒數要用的
    var secondsElapsed2:Int = 0,
    val handler2:Handler = Handler(Looper.getMainLooper()),
    var runnable2: Runnable = Runnable {}
)
