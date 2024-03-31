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

package org.tensorflow.lite.examples.poseestimation.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.SurfaceView
import android.widget.TextView
import kotlinx.coroutines.suspendCancellableCoroutine
import org.tensorflow.lite.examples.poseestimation.VisualizationUtils
import org.tensorflow.lite.examples.poseestimation.YuvToRgbConverter
import org.tensorflow.lite.examples.poseestimation.data.Person
import org.tensorflow.lite.examples.poseestimation.ml.MoveNetMultiPose
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.tensorflow.lite.examples.poseestimation.ml.TrackerType
import java.util.*
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.*
import java.util.Calendar

import org.tensorflow.lite.examples.poseestimation.data.standup_judge

class CameraSource(
    private var standup_judge_list:standup_judge,
    private val Stand_up_count_TextView: TextView,
    private val surfaceView: SurfaceView,
    private val listener: CameraSourceListener? = null
) {

    companion object {
        private const val PREVIEW_WIDTH = 640
        private const val PREVIEW_HEIGHT = 480

        /** Threshold for confidence score. */
        private const val MIN_CONFIDENCE = .5f
        private const val TAG = "Camera Source"
    }

    private val lock = Any()
    private var detector: PoseDetector? = null
    private var classifier: PoseClassifier? = null
    private var isTrackerEnabled = false
    private var yuvConverter: YuvToRgbConverter = YuvToRgbConverter(surfaceView.context)
    private lateinit var imageBitmap: Bitmap

    /** Frame count that have been processed so far in an one second interval to calculate FPS. */
    private var fpsTimer: Timer? = null
    private var frameProcessedInOneSecondInterval = 0
    private var framesPerSecond = 0

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = surfaceView.context
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** Readers used as buffers for camera still shots */
    private var imageReader: ImageReader? = null

    /** The [CameraDevice] that will be opened in this fragment */
    private var camera: CameraDevice? = null

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private var session: CameraCaptureSession? = null

    /** [HandlerThread] where all buffer reading operations run */
    private var imageReaderThread: HandlerThread? = null

    /** [Handler] corresponding to [imageReaderThread] */
    private var imageReaderHandler: Handler? = null
    private var cameraId: String = ""

    suspend fun initCamera() {
        camera = openCamera(cameraManager, cameraId)
        imageReader =
            ImageReader.newInstance(PREVIEW_WIDTH, PREVIEW_HEIGHT, ImageFormat.YUV_420_888, 3)
        imageReader?.setOnImageAvailableListener({ reader ->
            val image = reader.acquireLatestImage()
            if (image != null) {
                if (!::imageBitmap.isInitialized) {
                    imageBitmap =
                        Bitmap.createBitmap(
                            PREVIEW_WIDTH,
                            PREVIEW_HEIGHT,
                            Bitmap.Config.ARGB_8888
                        )
                }
                yuvConverter.yuvToRgb(image, imageBitmap)
                // Create rotated version for portrait display
                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)

                val rotatedBitmap = Bitmap.createBitmap(
                    imageBitmap, 0, 0, PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    rotateMatrix, false
                )
                processImage(rotatedBitmap)
                image.close()
            }
        }, imageReaderHandler)

        imageReader?.surface?.let { surface ->
            session = createSession(listOf(surface))
            val cameraRequest = camera?.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )?.apply {
                addTarget(surface)
            }
            cameraRequest?.build()?.let {
                session?.setRepeatingRequest(it, null, null)
            }
        }
    }

    private suspend fun createSession(targets: List<Surface>): CameraCaptureSession =
        suspendCancellableCoroutine { cont ->
            camera?.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(captureSession: CameraCaptureSession) =
                    cont.resume(captureSession)

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    cont.resumeWithException(Exception("Session error"))
                }
            }, null)
        }

    @SuppressLint("MissingPermission")
    private suspend fun openCamera(manager: CameraManager, cameraId: String): CameraDevice =
        suspendCancellableCoroutine { cont ->
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) = cont.resume(camera)

                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    if (cont.isActive) cont.resumeWithException(Exception("Camera error"))
                }
            }, imageReaderHandler)
        }

    fun prepareCamera() {
        for (cameraId in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (cameraDirection != null &&
                cameraDirection == CameraCharacteristics.LENS_FACING_FRONT) {
                this.cameraId = cameraId
                break // 找到前置摄像头后就跳出循环
            }

//            // We don't use a front facing camera in this sample.
//            val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
//            if (cameraDirection != null &&
//                cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
//            ) {
//                continue
//            }
//            this.cameraId = cameraId
        }
    }

    fun setDetector(detector: PoseDetector) {
        synchronized(lock) {
            if (this.detector != null) {
                this.detector?.close()
                this.detector = null
            }
            this.detector = detector
        }
    }

    fun setClassifier(classifier: PoseClassifier?) {
        synchronized(lock) {
            if (this.classifier != null) {
                this.classifier?.close()
                this.classifier = null
            }
            this.classifier = classifier
        }
    }

    /**
     * Set Tracker for Movenet MuiltiPose model.
     */
    fun setTracker(trackerType: TrackerType) {
        isTrackerEnabled = trackerType != TrackerType.OFF
        (this.detector as? MoveNetMultiPose)?.setTracker(trackerType)
    }

    fun resume() {
        imageReaderThread = HandlerThread("imageReaderThread").apply { start() }
        imageReaderHandler = Handler(imageReaderThread!!.looper)
        fpsTimer = Timer()
        fpsTimer?.scheduleAtFixedRate(
            object : TimerTask() {
                override fun run() {
                    framesPerSecond = frameProcessedInOneSecondInterval
                    frameProcessedInOneSecondInterval = 0
                }
            },
            0,
            1000
        )
    }

    fun close() {
        session?.close()
        session = null
        camera?.close()
        camera = null
        imageReader?.close()
        imageReader = null
        stopImageReaderThread()
        detector?.close()
        detector = null
        classifier?.close()
        classifier = null
        fpsTimer?.cancel()
        fpsTimer = null
        frameProcessedInOneSecondInterval = 0
        framesPerSecond = 0
    }

    // process image
//    private fun processImage(bitmap: Bitmap) {
//        val persons = mutableListOf<Person>()
//        var classificationResult: List<Pair<String, Float>>? = null
//
//        synchronized(lock) {
//            detector?.estimatePoses(bitmap)?.let {
//                persons.addAll(it)
//
//                // if the model only returns one item, allow running the Pose classifier.
//                if (persons.isNotEmpty()) {
//                    classifier?.run {
//                        classificationResult = classify(persons[0])
//                    }
//                }
//            }
//        }
//        frameProcessedInOneSecondInterval++
//        if (frameProcessedInOneSecondInterval == 1) {
//            // send fps to view
//            listener?.onFPSListener(framesPerSecond)
//        }
//
//        // if the model returns only one item, show that item's score.
//        if (persons.isNotEmpty()) {
//            listener?.onDetectedInfo(persons[0].score, classificationResult)
//        }
//        visualize(persons, bitmap)
//    }

    private fun processImage(bitmap: Bitmap) {
        // 创建一个矩阵用于翻转图像
        val matrix = Matrix().apply {
            preScale(1f, -1f, bitmap.width / 2f, bitmap.height / 2f)
        }
        // 创建翻转后的位图
        val flippedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        val persons = mutableListOf<Person>()
        var classificationResult: List<Pair<String, Float>>? = null

        synchronized(lock) {
            // 确保使用翻转后的位图进行姿态估计
            detector?.estimatePoses(flippedBitmap)?.let {
                persons.addAll(it)
                // if the model only returns one item, allow running the Pose classifier.
                if (persons.isNotEmpty()) {
                    classifier?.run {
                        classificationResult = classify(persons[0])
                    }
                }
            }
        }
        frameProcessedInOneSecondInterval++
        if (frameProcessedInOneSecondInterval == 1) {
            // send fps to view
            listener?.onFPSListener(framesPerSecond)
        }

        // 使用翻转后的位图进行可视化
        if (persons.isNotEmpty()) {
            listener?.onDetectedInfo(persons[0].score, classificationResult)
        }
        visualize(persons, flippedBitmap) // 确保这里也使用翻转后的位图
        Stand_up_count_fun(persons)
    }


    private fun Stand_up_count_fun(persons: List<Person>) {//2024/03/17
        if (persons[0].score > standup_judge_list.MLscore_threshold) {
            //val currentTime = Calendar.getInstance().time

            val RIGHT_HIP_x = persons[0].keyPoints[12].coordinate.x
            val RIGHT_HIP_y = persons[0].keyPoints[12].coordinate.y
            val RIGHT_KNEE_x = persons[0].keyPoints[14].coordinate.x
            val RIGHT_KNEE_y = persons[0].keyPoints[14].coordinate.y
            val RIGHT_ANKLE_x = persons[0].keyPoints[16].coordinate.x
            val RIGHT_ANKLE_y = persons[0].keyPoints[16].coordinate.y
            //var Previous_status
            val rvector1 = Vector(
                (RIGHT_HIP_x - RIGHT_KNEE_x).toDouble(),
                ((surfaceView.height - RIGHT_HIP_y) - (surfaceView.height - RIGHT_KNEE_y)).toDouble()
            )
            val rvector2 = Vector(
                (RIGHT_ANKLE_x - RIGHT_KNEE_x).toDouble(),
                ((surfaceView.height - RIGHT_ANKLE_y) - (surfaceView.height - RIGHT_KNEE_y)).toDouble()
            )
            val right_foot_angle = angleBetweenVectors(rvector1, rvector2)
//
            val LEFT_HIP_x = persons[0].keyPoints[11].coordinate.x
            val LEFT_HIP_y = persons[0].keyPoints[11].coordinate.y
            val LEFT_KNEE_x = persons[0].keyPoints[13].coordinate.x
            val LEFT_KNEE_y = persons[0].keyPoints[13].coordinate.y
            val LEFT_ANKLE_x = persons[0].keyPoints[15].coordinate.x
            val LEFT_ANKLE_y = persons[0].keyPoints[15].coordinate.y
            //var Previous_status
            val lvector1 = Vector(
                (LEFT_HIP_x - LEFT_KNEE_x).toDouble(),
                ((surfaceView.height - LEFT_HIP_y) - (surfaceView.height - LEFT_KNEE_y)).toDouble()
            )
            val lvector2 = Vector(
                (LEFT_ANKLE_x - LEFT_KNEE_x).toDouble(),
                ((surfaceView.height - LEFT_ANKLE_y) - (surfaceView.height - LEFT_KNEE_y)).toDouble()
            )
            val left_foot_angle = angleBetweenVectors(lvector1, lvector2)
//
//            Stand_up_count_TextView.text =
//                "41263"
//                right_foot_angle.toInt().toString() + " " + left_foot_angle.toInt().toString()
//
//
            if (standup_judge_list.index > standup_judge_list.current_index) {
                standup_judge_list.time_interval_list.add(right_foot_angle)
                standup_judge_list.time_interval_list.add(left_foot_angle)
                standup_judge_list.current_index += 2
                //standup_judge_list.time_interval_list
            } else {
                var time_interval_threshold_count: Int = 0;
                if (!standup_judge_list.Previous_status_standup) {
                    for (i in standup_judge_list.time_interval_list) {
                        if (i > standup_judge_list.stand_angle_threshold) {
                            time_interval_threshold_count++
                        }
                    }
                } else {
                    for (i in standup_judge_list.time_interval_list) {
                        if (i > standup_judge_list.sit_angle_threshold) {
                            time_interval_threshold_count++
                        }
                    }
                }


                var flage: Boolean = true

                if (time_interval_threshold_count / standup_judge_list.index >= standup_judge_list.time_interval_threshold) {
                    flage = true
                } else {
                    flage = false
                }

                if (flage && !standup_judge_list.Previous_status_standup) {
                    standup_judge_list.Count_Stand_up++
                    Stand_up_count_TextView.text = standup_judge_list.Count_Stand_up.toString()
                }

                if (flage) {
                    standup_judge_list.Previous_status_standup = true
                } else {
                    standup_judge_list.Previous_status_standup = false
                }

                standup_judge_list.time_interval_list.clear()
                standup_judge_list.current_index = 0
            }
//            Log.d("MyAppTag", "standup_judge_list : ${standup_judge_list.time_interval_list}")
//            Log.d("MyAppTag", "standup_judge_list : ${standup_judge_list.current_index}")
        }


    }
    data class Vector(val x: Double, val y: Double) {
        // 计算向量的模
        fun magnitude(): Double = sqrt(x*x + y*y)

        // 计算与另一个向量的点积
        infix fun dot(other: Vector): Double = this.x * other.x + this.y * other.y
    }

    // 计算两个向量之间的角度（以度为单位）
    fun angleBetweenVectors(v1: Vector, v2: Vector): Double {
        val dotProduct = v1 dot v2
        val magnitudeProduct = v1.magnitude() * v2.magnitude()
        val cosTheta = dotProduct / magnitudeProduct

        // acos 返回的是弧度，将其转换为度
        return Math.toDegrees(acos(cosTheta))
    }


    private fun visualize(persons: List<Person>, bitmap: Bitmap) {
//        for(i in persons){ //2024/03/03
//            Log.d("MyAppTag", "id : ${i.id}")
//            Log.d("MyAppTag", "keyPoints : ${i.keyPoints[12]}")
//            Log.d("MyAppTag", "keyPoints : ${i.keyPoints[14]}")
//            Log.d("MyAppTag", "keyPoints : ${i.keyPoints[16]}")
//            Log.d("MyAppTag", "score : ${i.score}")
//        }
//        Log.d("MyAppTag", "persons.size : ${persons.size}")
//        Log.d("MyAppTag", "keyPoints : ${persons[0].keyPoints[12]}")
//        Log.d("MyAppTag", "keyPoints : ${persons[0].keyPoints[14]}")
//        Log.d("MyAppTag", "keyPoints : ${persons[0].keyPoints[16]}")
//
//        Log.d("MyAppTag", "surfaceView.height : ${surfaceView.height}")
//        Log.d("MyAppTag", "surfaceView.height : ${surfaceView.width}")


        val outputBitmap = VisualizationUtils.drawBodyKeypoints(
            bitmap,
            persons.filter { it.score > MIN_CONFIDENCE }, isTrackerEnabled
        )

        val holder = surfaceView.holder
        val surfaceCanvas = holder.lockCanvas()
        surfaceCanvas?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
            surfaceView.holder.unlockCanvasAndPost(canvas)
        }
    }

    private fun stopImageReaderThread() {
        imageReaderThread?.quitSafely()
        try {
            imageReaderThread?.join()
            imageReaderThread = null
            imageReaderHandler = null
        } catch (e: InterruptedException) {
            Log.d(TAG, e.message.toString())
        }
    }

    interface CameraSourceListener {
        fun onFPSListener(fps: Int)

        fun onDetectedInfo(personScore: Float?, poseLabels: List<Pair<String, Float>>?)
    }
}
