package com.example.yolo_nas

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.round

class RectView(context: Context, attributeSet: AttributeSet) : View(context, attributeSet) {

    private var results: ArrayList<Result>? = null
    private lateinit var classes: Array<String>

    private val textPaint = Paint().apply {
        textSize = 60f
        color = Color.WHITE
    }

    // 라벨들의 배열 가져오기
    fun setClassLabel(classes: Array<String>) {
        this.classes = classes
    }

    // 화면의 크기에 맞게 바운딩 박스 크기 변환
    fun transformRect(results: ArrayList<Result>) {
        // scale 구하기
        val scaleY = height / DataProcess.INPUT_SIZE.toFloat()
        val scaleX = scaleY * 9f / 16f
        val realX = height * 9f / 16f
        val diffX = realX - width
        results.forEach {
            it.rectF.left = 0f.coerceAtLeast(it.rectF.left * scaleX - (diffX / 2f))
            it.rectF.right = width.toFloat().coerceAtMost(it.rectF.right * scaleX - (diffX / 2f))
            it.rectF.top = 0f.coerceAtLeast(it.rectF.top * scaleY)
            it.rectF.bottom = height.toFloat().coerceAtMost(it.rectF.bottom * scaleY)
        }

        this.results = results
    }

    // 그림 그리기
    override fun onDraw(canvas: Canvas?) {
        results?.forEach {
            canvas?.drawRect(it.rectF, findPaint(it.classIndex))
            canvas?.drawText(
                classes[it.classIndex] + ", " + round(it.score * 100) + "%",
                it.rectF.left + 10,
                it.rectF.top + 60,
                textPaint
            )
        }
        super.onDraw(canvas)
    }

    // 임의로 색상 지정
    private fun findPaint(classIndex: Int): Paint {
        val paint = Paint().apply {
            style = Paint.Style.STROKE      // 빈 사각형 그림
            strokeWidth = 10.0f             // 굵기 10
            strokeCap = Paint.Cap.ROUND     // 모서리는 뭉특하게
            strokeJoin = Paint.Join.ROUND   // 주위도 뭉특하게
            strokeMiter = 100f              // 뭉특한 정도
        }

        paint.color = when (classIndex) {
            0, 45, 18, 19, 22, 30, 42, 43, 44, 61, 71, 72 -> Color.WHITE
            1, 3, 14, 25, 37, 38, 79 -> Color.BLUE
            2, 9, 10, 11, 32, 47, 49, 51, 52 -> Color.RED
            5, 23, 46, 48 -> Color.YELLOW
            6, 13, 34, 35, 36, 54, 59, 60, 73, 77, 78 -> Color.GRAY
            7, 24, 26, 27, 28, 62, 64, 65, 66, 67, 68, 69, 74, 75 -> Color.BLACK
            12, 29, 33, 39, 41, 58, 50 -> Color.GREEN
            15, 16, 17, 20, 21, 31, 40, 55, 57, 63 -> Color.DKGRAY
            70, 76 -> Color.LTGRAY
            else -> Color.DKGRAY
        }
        return paint
    }
}