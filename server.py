import os
import io
import time
import base64
import json
from typing import Optional, List
import requests
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ==========================================
# 1. 초기 설정 및 모델 로드
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INVENTORY_FILE = os.path.join(BASE_DIR, "inventory.json")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-seg.pt")

# 만약 project_beta 내에 모델이 없으면 상위 project 폴더에서 복사 또는 자동 로드
if not os.path.exists(MODEL_PATH):
    parent_model = os.path.join(os.path.dirname(BASE_DIR), "project", "yolov8n-seg.pt")
    if os.path.exists(parent_model):
        MODEL_PATH = parent_model
    else:
        MODEL_PATH = "yolov8n-seg.pt"

print(f"[*] YOLOv8-Seg 모델을 로드하는 중... ({MODEL_PATH})")
yolo_model = YOLO(MODEL_PATH)
print("[+] YOLOv8-Seg 모델 로드 완료!")

# 인벤토리 파일 초기화
if not os.path.exists(INVENTORY_FILE):
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# 식약처 식중독 예측 API 캐시 (5분 TTL)
API_URL = "https://poisonmap.mfds.go.kr/api/risk.do"
poison_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 300  # 5분 (초)
}

app = FastAPI(title="Food Safety & YOLOv8 Spoilage System")

# CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 데이터 모델 정의
# ==========================================
class InventoryItem(BaseModel):
    name: str
    expiresAt: str
    registeredAt: Optional[str] = None
    spoilageScore: Optional[float] = None

class SpoilageRequest(BaseModel):
    imageBase64: str
    fruitHint: Optional[str] = None

# ==========================================
# 3. 헬퍼 함수
# ==========================================
def read_inventory() -> List[dict]:
    try:
        if os.path.exists(INVENTORY_FILE):
            with open(INVENTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"[!] 인벤토리 읽기 실패: {e}")
        return []

def write_inventory(items: List[dict]):
    with open(INVENTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def fetch_poison_data():
    now = time.time()
    if poison_cache["data"] and (now - poison_cache["timestamp"] < poison_cache["ttl"]):
        return poison_cache["data"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(API_URL, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    poison_cache["data"] = data
    poison_cache["timestamp"] = now
    return data

# ==========================================
# 4. YOLOv8 세그멘테이션 & 정밀 부패도 분석 엔진
# ==========================================
# COCO 데이터셋 과일/식품 영문명 -> 한국어 매핑
CLASS_KO_MAP = {
    "apple": "사과",
    "banana": "바나나",
    "orange": "오렌지 / 감귤",
    "broccoli": "브로콜리",
    "carrot": "당근",
    "hot dog": "핫도그",
    "pizza": "피자",
    "donut": "도넛",
    "cake": "케이크",
    "sandwich": "샌드위치"
}

DEFAULT_EXPIRY_DAYS = {
    "사과": 14,
    "바나나": 7,
    "오렌지 / 감귤": 10,
    "토마토": 8,
    "복숭아": 5,
    "딸기": 4,
    "포도": 7,
    "과일": 7
}

def analyze_fruit_with_yolov8(image_b64: str, fruit_hint: Optional[str] = None):
    # 1. Base64 이미지 디코딩 -> OpenCV BGR 이미지
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("이미지를 디코딩할 수 없습니다.")

    h, w = img_bgr.shape[:2]

    # 2. YOLOv8-Seg 모델 추론 수행
    results = yolo_model(img_bgr, verbose=False)
    result = results[0]

    detected_name = "과일"
    confidence = 0.90
    fruit_mask = np.zeros((h, w), dtype=np.uint8)
    fruit_detected = False

    # 세그멘테이션 마스크 추출
    if result.masks is not None and len(result.masks.data) > 0:
        # 가장 크거나 신뢰도가 높은 객체 탐색
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])

            # 과일/식품 또는 일반 객체 마스크 병합
            mask_data = result.masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
            bin_mask = (mask_resized > 0.5).astype(np.uint8)

            if np.sum(bin_mask) > (h * w * 0.01): # 최소 1% 이상 면적
                fruit_mask = np.bitwise_or(fruit_mask, bin_mask)
                if not fruit_detected or cls_name in CLASS_KO_MAP:
                    detected_name = CLASS_KO_MAP.get(cls_name, cls_name)
                    confidence = conf
                    fruit_detected = True

    # 마스크가 명확하지 않은 경우: 중앙 영역 또는 배경 제외 컨투어 추출
    if np.sum(fruit_mask) < (h * w * 0.02):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > (h * w * 0.05):
                cv2.drawContours(fruit_mask, [largest_cnt], -1, 1, thickness=cv2.FILLED)

    if np.sum(fruit_mask) == 0:
        fruit_mask = np.ones((h, w), dtype=np.uint8) # 전체 이미지 분석

    if fruit_hint and fruit_hint.strip():
        detected_name = fruit_hint.strip()

    # 3. 과일 마스크 내부 픽셀의 갈변 / 손상 / 부패 영역(HSV & LAB) 정밀 검출
    fruit_pixels_count = int(np.sum(fruit_mask))
    
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    # H: 0~180, S: 0~255, V: 0~255
    # L: 0~255 (명도), A: 녹색-적색, B: 청색-황색
    h_channel, s_channel, v_channel = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 손상 조건:
    # ① 흑갈색/갈변 반점: 낮은 명도(V < 90) + 과일 마스크 내부
    # ② 진한 갈색/변색: H in [5, 25], S > 60, V in [30, 160]
    # ③ 곰팡이/괴사 백색/흑색 반점: 극단적인 저명도 또는 국소 고대비 영역
    rot_mask_1 = (v_channel < 85) & (fruit_mask == 1)
    rot_mask_2 = (h_channel >= 5) & (h_channel <= 24) & (s_channel >= 65) & (v_channel <= 170) & (fruit_mask == 1)
    
    # 국소 그레이디언트 손상(무름/반점 텍스처)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    rot_mask_3 = (np.abs(laplacian) > 28) & (v_channel < 130) & (fruit_mask == 1)

    combined_rot_mask = np.zeros((h, w), dtype=np.uint8)
    combined_rot_mask[rot_mask_1 | rot_mask_2 | rot_mask_3] = 1

    # 모폴로지 연산으로 노이즈 제거 및 부패 영역 군집화
    kernel = np.ones((5, 5), np.uint8)
    combined_rot_mask = cv2.morphologyEx(combined_rot_mask, cv2.MORPH_OPEN, kernel)
    combined_rot_mask = cv2.dilate(combined_rot_mask, kernel, iterations=1)
    # 과일 영역 내로 제한
    combined_rot_mask = np.bitwise_and(combined_rot_mask, fruit_mask)

    rot_pixels_count = int(np.sum(combined_rot_mask))

    # 4. 부패도(%) 수치 산출
    if fruit_pixels_count > 0:
        raw_spoilage = (rot_pixels_count / fruit_pixels_count) * 100.0
        # 자연스러운 정밀 수치 보정 (0.0% ~ 100.0%)
        spoilage_percentage = round(min(100.0, max(0.0, raw_spoilage)), 1)
    else:
        spoilage_percentage = 0.0

    # 5. 상태 등급 및 행동 가이드 판정
    if spoilage_percentage < 5.0:
        grade = "신선함 (최적 상태)"
        grade_theme = "safe"
        action_guide = "표면 손상이 거의 없는 매우 신선한 상태입니다. 안심하고 섭취하세요."
    elif spoilage_percentage < 20.0:
        grade = "초기 변색 (섭취 권장)"
        grade_theme = "caution"
        action_guide = "자연스러운 갈변 또는 경미한 반점이 있습니다. 손상 부위만 살짝 도려내고 가급적 빨리 섭취하세요."
    elif spoilage_percentage < 50.0:
        grade = "부패 진행 (주의 필요)"
        grade_theme = "warning"
        action_guide = "부패 및 무름 현상이 상당 부분 진행되었습니다. 생식은 피하고 가열 조리하거나 폐기를 권장합니다."
    else:
        grade = "심각한 부패 (폐기 권고)"
        grade_theme = "danger"
        action_guide = "곰팡이 및 유해균 증식 위험이 매우 높습니다. 섭취하지 마시고 즉시 폐기하세요."

    # 6. AI 시각화 오버레이 이미지 생성 (과일 영역: 청록색 윤곽 / 부패 영역: 빨간색/주황색 하이라이트)
    overlay = img_bgr.copy()
    
    # 과일 영역 윤곽선 (Cyan)
    fruit_contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, fruit_contours, -1, (255, 200, 0), 2)

    # 부패 손상 영역 반투명 빨간색 채우기
    rot_color = np.zeros_like(img_bgr)
    rot_color[combined_rot_mask == 1] = [0, 0, 240] # BGR Red
    
    alpha = 0.5
    cv2.addWeighted(rot_color, alpha, overlay, 1 - alpha, 0, overlay)

    # 부패 영역 외곽선 (Bright Red)
    rot_contours, _ = cv2.findContours(combined_rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, rot_contours, -1, (0, 0, 255), 2)

    # 시각화 이미지 Base64 인코딩
    _, buffer = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    default_expiry = DEFAULT_EXPIRY_DAYS.get(detected_name, 7)

    return {
        "fruitName": detected_name,
        "confidence": f"{confidence * 100:.1f}%",
        "defaultExpiryDays": default_expiry,
        "spoilagePercentage": spoilage_percentage,
        "grade": grade,
        "gradeTheme": grade_theme,
        "actionGuide": action_guide,
        "analyzedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overlayImage": overlay_base64,
        "fruitAreaPixels": fruit_pixels_count,
        "damagedAreaPixels": rot_pixels_count
    }

# ==========================================
# 5. API 엔드포인트 라우팅
# ==========================================

# 1. 식약처 실시간 식중독 지수 조회
@app.get("/api/risk")
async def get_risk():
    try:
        data = fetch_poison_data()
        return data
    except Exception as e:
        print(f"[!] 식중독 API 실패: {e}")
        if poison_cache["data"]:
            return poison_cache["data"]
        raise HTTPException(status_code=500, detail=str(e))

# 2. 유통기한 인벤토리 목록 조회
@app.get("/api/inventory")
async def get_inventory():
    items = read_inventory()
    return {"success": True, "items": items}

# 3. 신규 식품 유통기한 등록
@app.post("/api/inventory", status_code=201)
async def add_inventory_item(item: InventoryItem):
    if not item.name or not item.expiresAt:
        raise HTTPException(status_code=400, detail="식품명과 유통기한은 필수입니다.")

    items = read_inventory()
    new_id = f"{int(time.time() * 1000)}_{os.urandom(3).hex()}"
    new_entry = {
        "id": new_id,
        "name": item.name.strip(),
        "registeredAt": item.registeredAt or time.strftime("%Y-%m-%d"),
        "expiresAt": item.expiresAt,
        "spoilageScore": item.spoilageScore
    }
    items.append(new_entry)
    write_inventory(items)
    return {"success": True, "item": new_entry, "items": items}

# 4. 식품 삭제
@app.delete("/api/inventory")
async def delete_inventory_item(id: str):
    if not id:
        raise HTTPException(status_code=400, detail="삭제할 항목의 id가 누락되었습니다.")

    items = read_inventory()
    filtered = [it for it in items if it.get("id") != id]
    write_inventory(filtered)
    return {"success": True, "items": filtered}

# 5. YOLOv8 과일 부패도 정밀 분석
@app.post("/api/analyze-spoilage")
async def analyze_spoilage(req: SpoilageRequest):
    if not req.imageBase64:
        raise HTTPException(status_code=400, detail="이미지 데이터가 없습니다.")

    try:
        analysis_result = analyze_fruit_with_yolov8(req.imageBase64, req.fruitHint)
        return {"success": True, **analysis_result}
    except Exception as e:
        print(f"[!] YOLOv8 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류: {str(e)}")

# 6. 메인 프론트엔드 서빙
@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_index():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h1>index.html 파일을 찾을 수 없습니다.</h1>", status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print("==================================================")
    print(" [YOLOv8 AI 기반 식중독 & 식품 위생 관리 시스템 Beta] ")
    print(f" 웹 브라우저 접속 주소: http://localhost:{port}")
    print(" 종료하려면 터미널에서 Ctrl+C를 누르세요.")
    print("==================================================")
    uvicorn.run(app, host="0.0.0.0", port=port)
