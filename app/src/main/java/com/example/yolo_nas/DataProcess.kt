package com.example.yolo_nas

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class DataProcess {

    lateinit var classes: Array<String>

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_SIZE = 640
        const val PIXEL_SIZE = 3
        const val FILE_NAME = "yolo_nas_s.onnx"
        const val LABEL_NAME = "coco.txt"
    }

    // coco label 불러오기
    fun loadLabel(assets: AssetManager) {
        // txt 파일 불러오기
        BufferedReader(InputStreamReader(assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = ArrayList<String>()
            while (reader.readLine().also { line = it } != null) {
                classList.add(line!!)
            }
            classes = classList.toTypedArray()
        }
    }

    // yolo nas 모델 불러오기
    fun loadModel(assets: AssetManager, filesDir: String) {
        // assets 안에 있는 파일 불러오기
        val outputFile = File("$filesDir/$FILE_NAME")
        assets.open(FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
    }

    // imageProxy -> bitmap
    fun imgToBmp(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val bitmap640 = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val matrix = Matrix()
        matrix.postRotate(90f)
        return Bitmap.createBitmap(bitmap640, 0, 0, INPUT_SIZE, INPUT_SIZE, matrix, true)
    }

    // bitmap -> floatBuffer
    fun bmpToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val imageSTD = 255.0f

        val cap = BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE
        val order = ByteOrder.nativeOrder()
        val buffer = ByteBuffer.allocateDirect(cap * Float.SIZE_BYTES).order(order).asFloatBuffer()

        val area = INPUT_SIZE * INPUT_SIZE
        val bitmapData = IntArray(area)     // 한 장의 사진에 대한 픽셀을 담을 배열
        bitmap.getPixels(bitmapData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // 배열에서 하나씩 가져와서 buffer에 담기
        for (i in 0 until INPUT_SIZE) {
            for (j in 0 until INPUT_SIZE) {
                val idx = INPUT_SIZE * i + j
                val pixelValue = bitmapData[idx]
                // pixel -> R, G, B 값 추출 & 양자화
                buffer.put(idx, Color.red(pixelValue) / imageSTD)
                buffer.put(idx + area, Color.green(pixelValue) / imageSTD)
                buffer.put(idx + area * 2, Color.blue(pixelValue) / imageSTD)
            }
        }
        buffer.rewind()
        return buffer
    }

    // 2차원 배열 output -> conf 임계값을 넘지 못한 배열들 제거 & nms 처리
    fun outputToPredict(output1: Array<*>, output2: Array<*>): ArrayList<Result> {
        val confidenceThreshold = 0.45f
        val results = ArrayList<Result>()
        val rows = output1.size

        for (i in 0 until rows) {
            // 80개의 라벨들 중 가장 높은 확률을 가진 라벨 및 확률
            val max = (output2[i] as FloatArray).withIndex().maxBy { it.value }
            val maxValue = max.value
            val maxIndex = max.index
            // 확률은 conf 임계값을 넘어야만 한다.
            if (maxValue > confidenceThreshold) {
                // 해당 인덱스의 xywh를 구한다.
                val x1 = (output1[i] as FloatArray)[0]
                val y1 = (output1[i] as FloatArray)[1]
                val x2 = (output1[i] as FloatArray)[2]
                val y2 = (output1[i] as FloatArray)[3]
                // 사각형은 화면 밖을 나갈 수 없으니 넘기면 최대치로 변경
                val rectF = RectF(
                    max(0f, x1), max(0f, y1),
                    min(INPUT_SIZE - 1f, x2), min(INPUT_SIZE - 1f, y2)
                )
                val result = Result(maxIndex, maxValue, rectF)
                results.add(result)
            }
        }
        return nms(results)
    }

    // 비 최대 억제 (nms)
    private fun nms(results: ArrayList<Result>): ArrayList<Result> {
        val list = ArrayList<Result>()

        for (i in classes.indices) {
            // 라벨들 중에서 가장 높은 확률값을 가졌던 라벨 찾기
            val pq = PriorityQueue<Result>(50) { o1, o2 ->
                o1.score.compareTo(o2.score)
            }
            val classResults = results.filter { it.classIndex == i }
            pq.addAll(classResults)

            //NMS 처리
            while (pq.isNotEmpty()) {
                // 큐 안에 속한 최대 확률값을 가진 class 저장
                val detections = pq.toTypedArray()
                val max = detections[0]
                list.add(max)
                pq.clear()

                // 교집합 비율 확인하고 50%넘기면 제거
                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    val rectF = detection.rectF
                    val iouThresh = 0.5f
                    if (boxIOU(max.rectF, rectF) < iouThresh) {
                        pq.add(detection)
                    }
                }
            }
        }
        return list
    }

    // 겹치는 비율 (교집합/합집합)
    private fun boxIOU(a: RectF, b: RectF): Float {
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    // 교집합
    private fun boxIntersection(a: RectF, b: RectF): Float {
        val w = overlap(
            (a.left + a.right) / 2f, a.right - a.left,
            (b.left + b.right) / 2f, b.right - b.left
        )
        val h = overlap(
            (a.top + a.bottom) / 2f, a.bottom - a.top,
            (b.top + b.bottom) / 2f, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0f else w * h
    }

    // 합칩합
    private fun boxUnion(a: RectF, b: RectF): Float {
        val i = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    // 겹치는 길이
    private fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = max(l1, l2)
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = min(r1, r2)
        return right - left
    }
}