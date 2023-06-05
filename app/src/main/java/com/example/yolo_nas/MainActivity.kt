package com.example.yolo_nas

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import java.util.Collections
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private lateinit var rectView: RectView
    private val dataProcess = lazy { DataProcess() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        rectView = findViewById(R.id.rectView)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setPermissions()
        load()
        setCamera()
    }

    // 카메라 켜기
    private fun setCamera() {
        // 카메라 제공 객체
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()
        // 전체 화면
        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        // 후면 카메라 설정
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        // 화면 비 16:9로 설정
        val resolutionSelector = ResolutionSelector.Builder().setAspectRatioStrategy(
            AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
        ).build()
        // 카메라로 부터 받아온 preview
        val preview = Preview.Builder().setResolutionSelector(resolutionSelector).build()
        // preview 를 화면에 보이기
        preview.setSurfaceProvider(previewView.surfaceProvider)
        // 화면 분석, 분석 중일땐 화면 대기가 아니라 계속 화면 새로 고침 분석이 끝나면 최신 사진을 다시 분석
        val analysis = ImageAnalysis.Builder().setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
        // 이미지 분석 메서드
        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }
        // 카메라의 수명을 메인 액티비티에 귀속
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    // 사진 분석
    private fun imageProcess(imageProxy: ImageProxy) {
        // YOLO_NAS_S : 0.41 ~ 0.46 초 소요, YOLO_NAS_S_QAT : 0.50 ~ 0.6 초 소요
        // 양자화 되는 것이 아니라 float 을 받아와서 int 형으로 변환 시키는 거라 오래 걸리는 듯...
        val bitmap = dataProcess.value.imgToBmp(imageProxy)
        val floatBuffer = dataProcess.value.bmpToFloatBuffer(bitmap)
        val inputName = ortSession.inputNames.iterator().next()
        // 모델의 입력 형태 [1 3 640 640] [배치 사이즈, 픽셀, 너비, 높이], 모델마다 다를 수 있음
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = ortSession.run(Collections.singletonMap(inputName, inputTensor))
        val output1 = (resultTensor.get(0).value as Array<*>)[0] as Array<*> // x, y, 너비, 높이
        val output2 = (resultTensor.get(1).value as Array<*>)[0] as Array<*> // 각 레이블 별 확률
        val results = dataProcess.value.outputToPredict(output1, output2)

        // 화면 표출
        rectView.transformRect(results)
        rectView.invalidate()

    }

    // 모델 불러오기
    private fun load() {
        dataProcess.value.loadModel(assets, filesDir.toString())
        dataProcess.value.loadLabel(assets)

        // 추론 객체
        ortEnvironment = OrtEnvironment.getEnvironment()
        // 모델 객체
        ortSession =
            ortEnvironment.createSession(
                filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME
            )

        // 라벨링 배열 전달
        rectView.setClassLabel(dataProcess.value.classes)
    }

    // 권한 확인
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == 1) {
            grantResults.forEach {
                if (it != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "권한을 허용 하지 않으면 사용할 수 없습니다!", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    // 권한 요청
    private fun setPermissions() {
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.CAMERA)

        permissions.forEach {
            if (ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions.toTypedArray(), 1)
            }
        }
    }
}