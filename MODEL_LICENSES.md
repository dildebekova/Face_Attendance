# Модели и лицензии
Код проекта — MIT (LICENSE). Это не распространяется на внешние библиотеки, исходники и веса.

- InsightFace: https://github.com/deepinsight/insightface — код MIT, публичные pretrained models только для некоммерческих исследований. Пилот в университете не автоматически считается исследованием. Для эксплуатационного внедрения запросите разрешение/лицензию: https://www.insightface.ai/ .
- Silent-Face-Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing — проверьте актуальные LICENSE, README и условия весов; сохраняйте уведомления upstream.
- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker — проверьте условия модели и SDK.
- ONNX Runtime: https://onnxruntime.ai/ ; OpenCV: https://opencv.org/ .

Веса не включены в ZIP. setup_models.py загружает их только после явного флага принятия исследовательских ограничений. Скачивает Silent-Face с зафиксированным commit b6d5f04ad78778917853b25c778acef6d5626d15, записывает commit и SHA256 весов. Не загружайте .pth из неизвестных источников: загрузчик использует PyTorch weights_only=True.
