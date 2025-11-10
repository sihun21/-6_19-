import os
from ultralytics import YOLO

# --- 설정 ---

# 🌟 [중요!] 이 MODEL_PATH 줄만 바꿔가며 3번 실행합니다.
# (1) Baseline 모델 (기본)
# MODEL_PATH = 'pothole_detection_final/run_roboflow_dataset_v2_boosted/weights/best.pt'
# (2) 실험 A 모델
MODEL_PATH = 'pothole_detection_final/run_CustomArch_ReLU_AdamW/weights/best.pt'
# (3) 실험 B 모델
# MODEL_PATH = 'pothole_detection_final/run_CustomArch_Wide_ReLU_AdamW/weights/best.pt'


# 벤치마킹에 사용할 고정된 비디오
SOURCE_TO_PREDICT = 'test_video.mp4' 
# --- ---

def main():
    # 1. 모델 파일이 있는지 확인
    if not os.path.exists(MODEL_PATH):
        print(f"오류: 모델 파일 '{MODEL_PATH}'를 찾을 수 없습니다.")
        return
    if not os.path.exists(SOURCE_TO_PREDICT):
        print(f"오류: 테스트 비디오 '{SOURCE_TO_PREDICT}'를 찾을 수 없습니다.")
        return

    # 2. 학습된 best.pt 모델 로드
    print(f"\n--- 벤치마킹 시작 ---")
    print(f"모델 로드 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 3. 예측 실행 (속도 측정)
    print(f"'{SOURCE_TO_PREDICT}' 파일로 속도 측정을 시작합니다...")
    
    results = model.predict(
        source=SOURCE_TO_PREDICT,
        imgsz=640,    # 🌟 1. 학습과 동일한 크기로 고정
        show=False,   # 🌟 2. 창을 띄우지 않음 (속도 측정)
        stream=False, # 🌟 3. 스트림 모드 끔 (평균 속도 측정)
        save=False,   # 🌟 4. 결과 저장 안 함 (속도만 측정)
        conf=0.25,    # (탐지 신뢰도 - 속도에 큰 영향 없음)
        device=0      # 🌟 GPU 사용 명시
    )

    # stream=False이므로, YOLO가 모든 프레임 처리를 완료하고
    # 자동으로 터미널 맨 마지막에 Speed: ... 로그를 출력합니다.
    
    print("--- 벤치마킹 완료 ---")
    print("터미널의 마지막 'Speed: ...' 로그에서 'inference' 시간을 확인하세요.")

if __name__ == '__main__':
    main()